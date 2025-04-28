import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = []
    for i in range(N):
        A.append([-u[i][0], -u[i][1], -1, 0, 0, 0, u[i][0]*v[i][0], u[i][1]*v[i][0],v[i][0]])
        A.append([0, 0, 0, -u[i][0], -u[i][1], -1, u[i][0]*v[i][1], u[i][1]*v[i][1],v[i][1]])
    A = np.array(A)
    
    # TODO: 2.solve H with A
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    h = 1.0 / np.sum(h) * h
    H = h.reshape((3,3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.arange(xmin, xmax)
    y = np.arange(ymin, ymax)

    X, Y = np.meshgrid(x, y)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    ones = np.ones_like(X_flat)
    coords = np.vstack([X_flat, Y_flat, ones])
    coords = coords.T

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        H_inv = np.linalg.inv(H)
        # Apply H_inv
        warped_coords = coords @ H_inv.T  # 注意是 H_inv的轉置

        # Normalize homogeneous coordinates
        warped_u = warped_coords[:, 0] / warped_coords[:, 2]
        warped_v = warped_coords[:, 1] / warped_coords[:, 2]

        # Reshape成影像形狀 (y,x)
        warped_u = warped_u.reshape((ymax - ymin, xmax - xmin))
        warped_v = warped_v.reshape((ymax - ymin, xmax - xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (
            (warped_u >= 0) & (warped_u < w_src) &
            (warped_v >= 0) & (warped_v < h_src)
        )

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        sample_u = np.round(warped_u[mask]).astype(int)
        sample_v = np.round(warped_v[mask]).astype(int)

        # 防止index超出範圍
        sample_u = np.clip(sample_u, 0, src.shape[1] - 1)
        sample_v = np.clip(sample_v, 0, src.shape[0] - 1)

        # TODO: 6. assign to destination image with proper masking
        dst_y, dst_x = np.where(mask)  # destination上的合法位置

        for c in range(ch):
            dst[dst_y, dst_x, c] = src[sample_v, sample_u, c]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        warped_coords = coords @ H.T  # (N,3) * (3,3)T --> (N,3)

        # Normalize
        warped_u = warped_coords[:, 0] / warped_coords[:, 2]
        warped_v = warped_coords[:, 1] / warped_coords[:, 2]

        # Reshape回成image形狀 (y,x)
        warped_u = warped_u.reshape((ymax - ymin, xmax - xmin))
        warped_v = warped_v.reshape((ymax - ymin, xmax - xmin))


        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (
            (warped_u >= 0) & (warped_u < w_dst) &
            (warped_v >= 0) & (warped_v < h_dst)
        )

        # TODO: 5.filter the valid coordinates using previous obtained mask
        warped_u_valid = warped_u[mask]
        warped_v_valid = warped_v[mask]

        dst_x = np.round(warped_u_valid).astype(int)
        dst_y = np.round(warped_v_valid).astype(int)

        # Source是從(X,Y)來的，所以src_x, src_y也是要用mask選
        src_x = X[mask]
        src_y = Y[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        valid_idx = (
            (dst_x >= 0) & (dst_x < w_dst) &
            (dst_y >= 0) & (dst_y < h_dst)
        )

        dst_x = dst_x[valid_idx]
        dst_y = dst_y[valid_idx]
        src_x = src_x[valid_idx]
        src_y = src_y[valid_idx]

        for c in range(ch):
            dst[dst_y, dst_x, c] = src[src_y, src_x, c]

    return dst 
