#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
import cython
from libc.math cimport isnan, NAN

#DOC
#python3 setup.py build_ext --inplace           ---> to create .so
#cython -a DTW_CCore.pyx                       ---> to create html check

#WARNINGS 
#does not work with more than 65000 points

cdef unsigned short U2undef = 65000
cdef float F4undef = 3e38
cdef float F4l = 1e20  # Large value. Normally not used, except to flag a special layer that absolutely need to be matched.

# Table distance entre grains depuis Lehning 2001 (normalise a 1, 0=parfait accord)
cdef float GRAIN_G[10][10]
#           X    1PP  2DF  3RG  4FC  5DH  6MF  7IF  8SH  9PPgp
GRAIN_G = [[0,   F4l, F4l, F4l, F4l, F4l, F4l, F4l, F4l, F4l], # ligne inutile pour idexage a partir 1
           [F4l, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8], # 1PP
           [F4l, 0.2, 0.0, 0.2, 0.6, 1.0, 1.0, 1.0, 1.0, 0.6], # 2DF
           [F4l, 0.5, 0.2, 0.0, 0.6, 0.9, 1.0, 0.0, 1.0, 0.5], # 3RG
           [F4l, 0.8, 0.6, 0.6, 0.0, 0.2, 1.0, 0.0, 1.0, 0.2], # 4FC
           [F4l, 1.0, 1.0, 0.9, 0.2, 0.0, 1.0, 0.0, 1.0, 0.3], # 5DH
           [F4l, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.2, 1.0, 1.0], # 6MF
           [F4l, 1.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0, 1.0], # 7IF
           [F4l, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], # 8SH
           [F4l, 0.8, 0.6, 0.5, 0.2, 0.3, 1.0, 1.0, 1.0, 0.0]] # 9PPgp


cdef float Distance_continue(float S, float T, float C):
    """
    Function that compute a distance between two values. Return (S-T)/C.
    """
    if(isnan(S) or isnan(T)):
        return NAN
    else:
        return abs(S - T) / C

cdef float Distance_grains(float S, float T):
    """
    Function that compute the distance between two grain types.
    Note that grain types are multiplexed (G1 + 10xG2).
    """
    cdef float dist_1, dist_2, dist_12, dist_21, dist_cross, dist_direc
    cdef int g_S0, g_S1, g_T0, g_T1
    # Demultiplexer les grains :
    g_S1 = int(S % 10)
    g_S0 = int((S - g_S1) / 10)
    g_T1 = int(T % 10)
    g_T0 = int((T - g_S1) / 10)

    # Calcul distance
    dist_1 = GRAIN_G[g_S0][g_T0]
    dist_2 = GRAIN_G[g_S1][g_T1]
    dist_12 = GRAIN_G[g_S0][g_T1]
    dist_21 = GRAIN_G[g_S1][g_T0]
    dist_cross = (dist_21+dist_12) / 2.
    dist_direc = (dist_1+dist_2) / 2.
    return min(dist_cross, dist_direc)

cdef float Distance(float[:] S, float[:] T, float[:,:] C, unsigned short M):
    """
    Function that compute the distance between two values of the profile.

    Basically, this function call function Distance_continue or Distance_grains
    and update the cost matrix C.
    """
    cdef float dist
    cdef float coeffs
    cdef float dd
    dist = 0
    coeffs = 0
    for i in range(0,M):
        if(C[i,2]==0):
            dd = Distance_continue(S[i], T[i], C[i, 1])
        else:
            dd = Distance_grains(S[i], T[i]) 
        if(not isnan(dd)):
            dist += C[i,0] * dd
            coeffs +=C[i,0]
    if(coeffs==0):
        dist = 0
    else:
        dist = dist/coeffs
    return dist

cdef float coutdecal(int i, int j, float COUTDECAL):
    return abs(i-j)*COUTDECAL


cpdef float DTW_multi(float[:, :] S, float[:, :] T, float[:,:] C, float[:,:] D, char[:,:] B, unsigned short [:,:] ID, int partial, float COUTDECAL):
    '''
    :param S: 2D numpy array to compare. S is the reference one.
    :param T: 2D numpy array to compare. T is the signal to map with S (profile to be matched).
    :param C: Information for Distance.
        Contains for each variable, a length-3 array :
         - Coefficient
         - Amplitude of variable (for renormalisation)
         - isGrain (0 or 1)
    :param D: Array of distance, filled by this function
    :param B: Array of backtrace, filled by this function
    :param ID: Array of result, filled by this function
    :param partial: max number of points not associated at the end of T (or S, I don't remember)
    '''
    
    cdef unsigned short N = S.shape[1]
    cdef unsigned short M = S.shape[0]
    cdef int i,j,pi,best_i,best_j,max_S,max_T
    cdef float d0,d1,d2,c2,c4,c5,best_dist,cd
    cdef unsigned short nnan
    
    max_S = N
    max_T = N
    
    for i in range(0,N):
        nnan = 0
        for j in range(0,M):
            if(isnan(S[j,i])):
                nnan += 1
        if nnan == M:
            max_S = i
            break
    
    for i in range(0,N):
        nnan = 0
        for j in range(0,M):
            if(isnan(T[j,i])):
                nnan += 1
        if nnan == M:
            max_T = i
            break

    if max_S < 2 or max_T <2 :
        print("Start is nan. Check your data")
    
    #initialization of ID
    for i in range(1,2*N-1):
        ID[i,0] = U2undef
        ID[i,1] = U2undef
    
    #Creating D distance array and backtrace array
    for i in range(1,N):
        #i=0,1
        D[0,i] = F4undef
        D[1,i] = F4undef
        
        #j=0,1
        D[i,0] = F4undef
        D[i,1] = F4undef
    
    #for i in range(0,M):
    #    if(isnan(S[i, 0]) or isnan(T[i, 1]) or isnan(S[i, 0]) or isnan(T[i, 1])):
    #        D[0,0] = 0
    #        D[1,1] = 0
    #        B[1,1] = 3
    #        print('Start is nan. Check your data')
    else:
        d0 = Distance(S[:,0], T[:,0], C, M)
        D[0,0] = d0
        B[0,0] = 18
        
        d0 = Distance(S[:,1], T[:,1], C, M)
        D[1,1] = d0
        B[1,1] = 3
    
    #i>1 and j>1
    for i in range(2,max_S):       
        for j in range(2,max_T):

            cd = coutdecal(i, j, COUTDECAL)
            d0 = Distance(S[:,i], T[:,j], C, M) + cd
            d1 = Distance(S[:,i-1], T[:,j], C, M) + cd
            d2 = Distance(S[:,i], T[:,j-1], C, M) + cd

            c2 = D[i-1,j-1] 
            c4 = D[i-2,j-1] + d1
            c5 = D[i-1,j-2] + d2
            
            if(c4<c2):
                
                if(c4<c5):
                    D[i,j] = c4 + d0
                    B[i,j] = 4
                else:
                    D[i,j] = c5 + d0
                    B[i,j] = 5
                    
            elif(c2<c5):
                D[i,j] = c2 + d0
                B[i,j] = 2
                
            else:
                D[i,j] = c5 + d0
                B[i,j] = 5

    #Backtracing
    if partial>0:
        best_i = max_S-1
        best_j = max_T-1
        best_dist = F4undef
        
        for i in range(max_S-partial,max_S):
            if D[i,max_T-1] < best_dist:
                best_i = i
                best_j = max_T-1
                best_dist = D[best_i,best_j]
                
        for j in range(max_T-partial,max_T):
            if D[max_S-1,j] < best_dist:
                best_i = max_S-1
                best_j = j
                best_dist = D[best_i,best_j]
            
    else:
        best_i = max_S-1
        best_j = max_T-1
              
    i = best_i
    j = best_j
    
    pi = 0
    ID[pi,0] = i
    ID[pi,1] = j

    while(i>0):
        if  (B[i,j]==1):
            i -= 1
            
        elif(B[i,j]==2):
            i -= 1
            j -= 1
            
        elif(B[i,j]==3):
            j -= 1
            
        elif(B[i,j]==4):
            i -= 2
            j -= 1
            
        elif(B[i,j]==5):
            i -= 1
            j -= 2
        else:
            #print (i,j,B[i,j],max_S,max_T)
            break
        
        pi += 1
        ID[pi,0] = i
        ID[pi,1] = j

    return D[best_i,best_j]
