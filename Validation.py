import numpy as np

def CalibrationValidation(intrinsicL, intrinsicR, rot, trans, essential, fundamental):
    # 3 premières colonnes de Matrice de projection = produit de M_int et 3 première colonne de M_extr(Matrice de rotation)
    
    # Matrice intrinsèque doit être une matrice triangulaire supérieure
    print("======== CALIBRATION VALIDATION ===============")
    isIntrinsicRTriangular = np.allclose(intrinsicR, np.triu(intrinsicR))
    isIntrinsicLTriangular = np.allclose(intrinsicL, np.triu(intrinsicL))
    print("Is left instrinsic matrix a upper triangular ? ", isIntrinsicRTriangular)
    print("Is right instrinsic matrix a upper triangular ? ", isIntrinsicLTriangular)

    # Matrice de rotation doit être une matrice orthonormale
    I = np.identity(rot.shape[0])
    isRotOrthogonal = np.allclose(np.matmul(rot, np.transpose(rot)), I)
    isRotOrthonormal = isRotOrthogonal and np.allclose(np.linalg.inv(rot), rot.T) 
    print("Is Rotation Matrix orthonormal ? ", isRotOrthonormal)

    # Singular value decomposition de la matrice essentielle doit nous donner matrice Rotation & Matrice Translation
    # Matrice essentielle = M_intr_gauche * Mat_fond * M_intr_droite

    # m_ess = np.matmul(intrinsicL, np.matmul(fundamental, intrinsicR))
    # print(m_ess)
    # print("Does the essential matrix = intrisincLeft * fundamental * intrinsixR ? ", (np.allclose(m_ess, essential)))

def MatchingValidation(ptsR, ptsL, linesL, linesR, fundamental):
    # Le point trouvé à gauche * Matrice fondamentale * point trouvé à droite = 0 (epipolar constraint)
    # point gauche doit être sur la même ligne que epipole gauche (même chose pour point droite et épipole droite)
    print("======== MATCHING VALIDATION ===============")

    epipoleConstraint = True
    ptLOnLine = True
    ptROnLine = True
    eps = 0.01
    index = 0
    for ptL, ptR in zip(ptsL, ptsR):
        l = np.array([ptL[0], ptL[1], 1])
        r = np.array([ptR[0], ptR[1], 1])
        lineL = linesL[index]
        lineR = linesR[index]
        prod1 = np.matmul(fundamental, r)
        result = np.dot(l, prod1)
        if (np.absolute(result) < eps):
            epipoleConstraint = False
            break

        # ax + by + c = 0 équation de la droite
        resultL = lineL[0] * ptL[0] + lineL[1] * ptL[1] + lineL[2]
        resultR = lineR[0] * ptR[0] + lineR[1] * ptR[1] + lineR[2]

        if (np.absolute(resultL) < eps):
            ptLOnLine = False
            break
        
        if (np.absolute(resultR) < eps):
            ptROnLine = False
            break
        index = index + 1

    print("Epipolar constraint respected ? ", epipoleConstraint)
    print("Left points are on epipolar line ? ", ptLOnLine)
    print("Right points are on epipolar line ? ", ptROnLine)

def DepthValidation():
    pass