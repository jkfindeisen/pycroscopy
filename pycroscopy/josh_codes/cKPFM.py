def fit_cKPFM_linear(bias_vec_r,resp_mat_vsr):

    fit_slope = np.zeros(resp_mat_vsr.shape)
    fit_yintercept = np.zeros(resp_mat_vsr.shape)
    jCPD_mat = np.zeros(resp_mat_vsr.shape)

    for row in range(resp_mat_vsr.shape[0]):
        for col in range(resp_mat_vsr.shape[1]):
            for vw in range(resp_mat_vsr.shape[2]):

                idx = np.isfinite(resp_mat_vsr[col, row, vw, :])
                fit_coeff = np.polyfit(
                    bias_vec_r[idx],
                    resp_mat_vsr[col, row, vw, idx], 1)
                fit_slope[col, row, vw] = fit_coeff[0]
                fit_yintercept[col, row, vw] = fit_coeff[1]
                jCPD_mat[col, row, vw] = -fit_coeff[1] / fit_coeff[0]

    return(fit_slope, fit_yintercept, jCPD_mat)
