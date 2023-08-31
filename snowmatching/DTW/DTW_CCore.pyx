#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3

import numpy as np
import cython
# from cython.parallel import threadid, prange

from libc.math cimport isnan, NAN, fabs
#DOC
#python3 setup.py build_ext --inplace           ---> to create .so
#cython -a DTW_CCore.pyx                       ---> to create html check

#WARNINGS
#does not work with more than 65000 points

cdef unsigned short U2undef = 65000
cdef float F4undef = 3e38
cdef float F4l = 1e20  # Large value. Normally not used, except to flag a special layer that absolutely need to be matched.

# Distance array between grains, inspired from Lehning 2001 (normalized to 1, 0=perfect agreement) and published in Viallon-Galinier, 2021
cdef float GRAIN_G[10][10]
#           X    1PP  2DF  3RG  4FC  5DH  6MF  7IF  8SH  9PPgp
GRAIN_G = [[0,   F4l, F4l, F4l, F4l, F4l, F4l, F4l, F4l, F4l], # X
           [F4l, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8], # 1PP
           [F4l, 0.2, 0.0, 0.2, 0.6, 1.0, 1.0, 1.0, 1.0, 0.6], # 2DF
           [F4l, 0.5, 0.2, 0.0, 0.6, 0.9, 1.0, 0.0, 1.0, 0.5], # 3RG
           [F4l, 0.8, 0.6, 0.6, 0.0, 0.2, 1.0, 0.0, 1.0, 0.2], # 4FC
           [F4l, 1.0, 1.0, 0.9, 0.2, 0.0, 1.0, 0.0, 1.0, 0.3], # 5DH
           [F4l, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.2, 1.0, 1.0], # 6MF
           [F4l, 1.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0, 1.0], # 7IF
           [F4l, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], # 8SH
           [F4l, 0.8, 0.6, 0.5, 0.2, 0.3, 1.0, 1.0, 1.0, 0.0]] # 9PPgp


cdef float Distance_continue(float S, float T, float C) nogil:
    """
    Function that compute a distance between two values. Return (S-T)/C.
    """
    if(isnan(S) or isnan(T)):
        return NAN
    else:
        return fabs(S - T) / C

cdef float Distance_grains(float S, float T) nogil:
    """
    Function that compute the distance between two grain types.
    Note that grain types are multiplexed (G1 + 10xG2).
    """
    cdef float dist_1, dist_2, dist_12, dist_21, dist_cross, dist_direc
    cdef int g_S0, g_S1, g_T0, g_T1
    # Demultiplex grains :
    g_S1 = int(S % 10)
    g_S0 = int((S - g_S1) / 10)
    g_T1 = int(T % 10)
    g_T0 = int((T - g_S1) / 10)

    # Computation of distance
    dist_1 = GRAIN_G[g_S0][g_T0]
    dist_2 = GRAIN_G[g_S1][g_T1]
    dist_12 = GRAIN_G[g_S0][g_T1]
    dist_21 = GRAIN_G[g_S1][g_T0]
    dist_cross = (dist_21+dist_12) / 2.
    dist_direc = (dist_1+dist_2) / 2.
    return min(dist_cross, dist_direc)

cdef float Distance(float[:] S, float[:] T, float[:,:] C, unsigned short M) nogil:
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
        dist = dist / coeffs
    return dist

cdef float coutdecal(int i, int j, float COUTDECAL) nogil:
    return fabs(i-j) * COUTDECAL


cpdef float DTW_multi(float[:, :] S, float[:, :] T, float[:,:] C, float[:,:] D, char[:,:] B, unsigned short [:,:] ID, int partial, float COUTDECAL) nogil:
    '''
    :param S: 2D numpy array to compare. S is the signal to map to T (profile to be matched).
    :type S: numpy array shape (n_profiles, n_variables, n_points)
    :param T: 2D numpy array to compare. T is the reference.
    :param C: Information for Distance computation (weights).
        Contains for each variable, a length-3 array :
         - Coefficient
         - Amplitude of variable (for renormalisation)
         - isGrain (0 or 1)
    :param D: Array of distance, filled by this function
    :param B: Array of backtrace, filled by this function
    :param ID: Array of result, filled by this function
    :param partial: max number of points not associated at the end of T (or S, I don't remember)
    :param COUTDECAL: Cost associated with layer displacement.
    :return: 
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
        # print("Start is nan. Check your data")
        return 1

    # initialization of ID
    for i in range(1,2*N-1):
        ID[i,0] = U2undef
        ID[i,1] = U2undef

    # Creating D distance array and backtrace array
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

    # i>1 and j>1
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

    # Backtracing
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
            break

        pi += 1
        ID[pi,0] = i
        ID[pi,1] = j

    return D[best_i,best_j]


cpdef DTW_set(float[:, :, :] Ss,
              int n_ite,
              float[:, :] C,
              int partial, float COUTDECAL,
              int n_thr = 1):
    '''
    Auto-fit of a set of profiles to determine a mean profile.

    NB : Such algorithm can progressively introduce a bias with an overall displacement of all the layers,
    in particular when no COUTDECAL is used. The mean displacement have to be corected as a post-processing.
    (See bias correction in usefull.py)

    NB : Currently, variable sshould be continuous variables (not grain shape, for instance) as averaging
    would not be propermy defined for discrete variables.

    :param Ss: 3D numpy array of the set of profiles to compare.
               Dimensions : number of profiles, length of the profiles, number of variables
    :param n_ite: Number of iterations for the mean computation
    :param C: Information for Distance comptation (weigths).
        Contains for each variable, a length-3 array :
         - Coefficient
         - Amplitude of variable (for renormalisation)
         - isGrain (0 or 1)
    :param partial: See DTW_multi documentation.
    :param COUTDECAL: See DTW_multi documentation.
    :param n_thr: Number of threads (more threads require more memory consumption)
    '''
    cdef int n_pro = Ss.shape[0]  # Number of profiles
    cdef int n_var = Ss.shape[1]  # Number of variables (features)
    cdef int n_poi = Ss.shape[2]  # Number of points
    cdef int n_pat = 2 * n_poi - 1

    cdef float[:, :, :] Ds = np.zeros((n_thr, n_poi, n_poi), dtype='float32')
    cdef char[:, :, :] Bs = np.zeros((n_thr, n_poi, n_poi), dtype='i1')
    cdef unsigned short[:, :, :] IDs = np.zeros((n_pro, n_pat, 2), dtype='u2')

    cdef float[:, :] M = np.zeros((n_var, n_poi), dtype='float32')  # Mean sequence
    cdef float[:, :] M_tmp = np.zeros((n_var, n_poi), dtype='float32')  # Temporary mean sequence
    cdef float[:, :] ok = np.zeros((n_var, n_poi), dtype='float32')  # Counter of points
    cdef int[:] max_Ss = np.zeros(n_pro, dtype='int32')  # Counter of max not nan

    cdef int max_M
    cdef int i_pro, i_poi, i_var, i_pat, i_S, i_M, i_thr, i_ite
    cdef double d_tot
    cdef float d0

    # Determination of the length of each profile
    for i_pro in range(n_pro):
        for i_poi in range(n_poi):
            found_val = False
            for i_var in range(n_var):
                if not isnan(Ss[i_pro, i_var, i_poi]):
                    break
            max_Ss[i_pro] = i_poi
            if found_val is False:
                break

    # Length of the mean: maximum length across profiles
    max_M = np.max(max_Ss)

    # Initialization of mean M
    M = np.nanmean(Ss, axis=0)

    # Iterations:
    #  - Matching from original profile to M
    #  - Recompute M
    for i_ite in range(n_ite):
        # Initialization of M_tmp and ok
        M_tmp[:, :] = 0
        ok[:, :] = 0

        # Initialization of total_distance
        d_tot = 0

        # Compute the DTW of each profile against mean
        for i_pro in range(n_pro):
        # for i_pro in prange(n_pro, nogil=True, num_threads=n_thr):

            # For one profile
            i_thr = 0  # threadid()
            # Compute DTW between profile and previous mean
            d0 = DTW_multi(Ss[i_pro], M,
                           C,
                           Ds[i_thr], Bs[i_thr], IDs[i_pro],
                           partial, COUTDECAL)
            d_tot += d0

            # Prepare next mean (compute the sum by
            # aggregation at each step. this way, it
            # is not necessary to store IDs table !)
            for i_pat in range(n_pat):
                i_S = IDs[i_pro, i_pat, 0]
                i_M = IDs[i_pro, i_pat, 1]
                if i_S == U2undef or i_M == U2undef:
                    break

                for i_var in range(n_var):
                    if not isnan(Ss[i_pro, i_var, i_S]):
                        M_tmp[i_var, i_M] += Ss[i_pro, i_var, i_S]
                        ok[i_var, i_M] += 1

                if i_S == 0 and i_M == 0:
                    break

        # Finalize averaging computation
        for i_var in range(n_var):
            for i_poi in range(n_poi):
                if(ok[i_var, i_poi] > 0):
                    M[i_var, i_poi] = M_tmp[i_var, i_poi] / ok[i_var, i_poi]
                elif i_poi < n_poi - 1 and ok[i_var, i_poi + 1] > 0:
                    M[i_var, i_poi] = 0.5 * (M[i_var, i_poi - 1] + M_tmp[i_var, i_poi + 1] / ok[i_var, i_poi + 1])
                else:
                    M[i_var, i_poi] = NAN  # M[i_poi-1]#0.5 * (M[i_point-1] + M_n[i_point+1] / IDC[i_point+1])

        # Some smoothing
        # Removed, if you want a smoothing, do a post-processing !
        #for i_var in range(n_var):
        #    for i_poi in range(2, n_poi - 2):
        #        if not isnan(M[i_var, i_poi - 1]) and not isnan(M[i_var, i_poi]) and not isnan(M[i_var, i_poi + 1]):
        #            M[i_var, i_poi] = (1.0 * ok[i_var, i_poi -2] * M[i_var, i_poi - 2] +
        #                               1.0 * ok[i_var, i_poi - 1] * M[i_var, i_poi - 1] +
        #                               1.0 * ok[i_var, i_poi] * M[i_var, i_poi] +
        #                               1.0 * ok[i_var, i_poi + 1] * M[i_var, i_poi + 1] +
        #                               1.0 * ok[i_var, i_poi + 2] * M[i_var, i_poi + 2]) / \
        #                               (1.0 * ok[i_var, i_poi - 2] + 1.0 * ok[i_var, i_poi - 1] +
        #                                1.0 * ok[i_var, i_poi] +
        #                                1.0 * ok[i_var, i_poi + 1] +
        #                                1.0 * ok[i_var, i_poi + 2])

    return np.copy(M), np.copy(IDs)


