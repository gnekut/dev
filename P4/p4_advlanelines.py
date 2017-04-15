import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


'''
# VISUALIZATION CLIPBOARD

'''
# ---------- calibration ----------
def camera_cal(img_paths, pts):
	'''
	Function:
	Calculate camera calibration matrix and distortion coefficients based upon the provided checkerboard image patterns.

	Inputs: 
	[img_paths: List of image files to calibrate camera]
	[pts: Predefined number of points in the calibration image.]

	Output:
	[cal_mtx: Camera calibraton matrix]
	[dist_coeff: Camera distortion coefficients]
	'''

	# Iterate through images to gather object/image points
	objpts = np.zeros((pts[0]*pts[1],3), np.float32)
	objpts[:,:2] = np.mgrid[0:pts[0],0:pts[1]].T.reshape(-1,2)
	imgpts_a, objpts_a, img_ind = [], [], []
	for i in img_paths:
		img = cv2.imread(i)
		img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		found_pts, imgpts = cv2.findChessboardCorners(img_g, pts, None)
		if found_pts:
			imgpts_a.append(imgpts)
			objpts_a.append(objpts)
			# plt.show(plt.imshow(cv2.drawChessboardCorners(img, pts, imgpts, found_pts)))
	# Calibrate camera and return calibration matrix and distortion coefficients
	ret, cal_mtx, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(objpts_a, imgpts_a, img_g.shape[::-1], None, None)
	return cal_mtx, dist_coeff

# ----------- perspective transforms ---------
def persepctive(img, edges=None):
	'''
	Function:
	To change a camera image of a certain perspective to a predefined birds-eye view

	Inputs:
	[img: Image input of any number of channels and/opr size]
	[edges: New projection coordinated. If not defined for the input the output will yield the same inputs]

	Outputs:
	[img_warp: Warped input image from a birds-eye view perspective. Calibrarted by using the staight line test images.]
	[M: Transformation matrix. For any given image input of the same dimensiomns/camera/perspective, the matrix will map x --> y]
	[Minv: Inverse transformation matrix. From a birds-eye view, map the newly defined image, or pixels, back to the original perspective. The matrix will map y --> x]
	'''

	Y, X = img.shape
	offset = (250, 10)
	if edges == None:
		edges = {'b1':X, 'b2':X, 'h1':Y, 'h2':0}
	# [TL, TR, BR, BL] (y,x)
	src = np.float32([[(X-edges['b2'])/2, Y-edges['h2']], [(X+edges['b2'])/2, Y-edges['h2']], [(X-edges['b1'])/2, Y-edges['h1']], [(X+edges['b1'])/2, Y-edges['h1']]])
	dst = np.float32([[offset[0], offset[1]], [X-offset[0], offset[1]], [offset[0], Y-offset[1]], [X-offset[0], Y-offset[1]]])

	M = cv2.getPerspectiveTransform(src, dst)
	Minv = cv2.getPerspectiveTransform(dst, src)
	img_warp = cv2.warpPerspective(img, M, (X,Y), flags=cv2.INTER_LINEAR)
	return img_warp, M, Minv

# ------------ color and gradient thresholds ------------
def channel_threshold(img_bgr, channel='S', threshold=(0,255)):
	'''
	Function:
	Filter an image to one of six channels (B,G,R,H,L,S). More can be added in the event that other resolutions are needed.

	Inputs:
	[img_bgr: Three channel image with indices corresponding to blue, green and red]
	[channel: Channel selection from those available in the definition. Acceptable inputs (BGRHLS)]
	[threshold: Tuple (min, max). Channel color thresholds (8-bit). Acceptable inputs (0-255)]

	Outputs:
	[image: Filtered channel image]

	'''
	ch_key = {'B':0, 'G':1, 'R':2, 'H':3, 'L':4, 'S':5}
	img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
	img_ch = np.dstack((img_bgr,img_hls))[:,:,ch_key[channel]]
	image = np.zeros_like(img_ch)
	image[(img_ch > threshold[0]) & (img_ch <= threshold[1])] = 1
	return image
def abs_sobel_threshold(img_bgr, orient='x', sobel_kernel=3, threshold=(0, 255)):
	'''
	Function:
	Filter grayscale image pixels falling between the defined directional gradient thresholds.
	
	Inputs:
	[img_bgr: Three channel image with indices corresponding to blue, green and red]
	[orient: Sobel derivative with respect to x/y]
	[sobel_kernel: NxN size of derivative matrix]
	[threshold: Tuple (min, max). Grayscale threshold for gradient values]

	Outputs:
	[grad_binary: Binary image mapping (0,1) of pixels that meet the directional gradient threshold.]
	'''

	image_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	sobel = cv2.Sobel(image_g, cv2.CV_64F, 1 if orient == 'x' else 0, 1 if orient == 'y' else 0, ksize=sobel_kernel)
	sobel_abs = np.absolute(sobel)
	scale = np.uint8(255*sobel_abs/np.max(sobel_abs))
	grad_binary = np.zeros_like(scale)
	grad_binary[(scale >= threshold[0]) & (scale <= threshold[1])] = 1
	return grad_binary
def mag_threshold(img_bgr, sobel_kernel=3, threshold=(0, 255)):
	'''
	Function: 
	Filter grayscale image pixels with a gradient magnitude (sqrt(dp/dx^2+dp/dy^2)) within the defined thresholds.
	
	Inputs:
	[img_bgr: Three channel image with indices corresponding to blue, green and red]
	[sobel_kernel: NxN size of derivative matrix]
	[threshold: Tuple (min, max). Grayscale threshold for gradient values]

	Outputs:
	[mag_binary: Binary image mapping (0,1) of pixels that meet the magnitude gradient threshold.]
	'''

	image_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(image_g, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(image_g, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	mag = np.sqrt(np.square(sobelx) + np.square(sobely))
	scale = np.uint8(255*mag/np.max(mag))
	mag_binary = np.zeros_like(scale)
	mag_binary[(scale >= threshold[0]) & (scale <= threshold[1])] = 1
	return mag_binary
def dir_threshold(img_bgr, sobel_kernel=3, threshold=(0, np.pi/2)):
	'''
	Function:
	Filter grayscale image pixels with a gradient direction (arctan(x/y)) within the defined thresholds. Help to detect gradients that are within a certain orientation (looking for vertical lane lines)

	Inputs:
	[img_bgr: Three channel image with indices corresponding to blue, green and red]
	[sobel_kernel: NxN size of derivative matrix]
	[threshold: Tuple (min, max). Grayscale threshold for gradient values]

	Outputs:
	[dir_binary: Binary image mapping (0,1) of pixels that meet the directional gradient threshold.]
	'''

	image_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	sobelx_abs = np.absolute(cv2.Sobel(image_g, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
	sobely_abs = np.absolute(cv2.Sobel(image_g, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
	grad_dir = np.arctan2(sobely_abs, sobelx_abs)
	dir_binary = np.zeros_like(grad_dir)
	dir_binary[(grad_dir >= threshold[0]) & (grad_dir <= threshold[1])] = 1
	return dir_binary

# ------------ line finder --------------
def line_find(img_warp, line_l, line_r, win_cnt=10, x_mrg=100, pxl_cnt_min=50, frame_history=3):

	'''
	Function:
	Determine the polynomial best fit line from a projected, thresholded image of lane lines. Inital line determination uses a moving window approach, starting at a base point (xlb_/xrb_) where the the concentration of valid pixels is highest in the bottom quarter of the image. A list is maintained that contains all pixels that lie within a predefined margin (x_marg) to either side of the base point within each y-window. If enough pixels are found within the window, a new center is calculated that will serve as the base for the next window. This process continues for the total number of input windows (win_cnt, greater number of windows, better line resolution but caution of noise and performance). If the previous picture frames yielded an accurate estimate of the lane line (CRITERIA TBD), then the calculation of the pixels/polynomial in the new frame should use that as a starting point for where to search in the image since there is continuity of the lane lines from the previous N-frames and the current one.

	Inputs:
	[img_warp: Perspective warped, binary thresholded image.]
	[line_l: Left lane line parameters (object) containing information about previous line detections.]
	[line_r: Right lane line parameters (object) containing information about previous line detections.]
	[win_cnt: Number of y-window steps to segment lane line search.]
	[x_mrg: Margin to each side of the window center or polynomial lane line.]
	[pxl_cnt_min: Minimum number of pixels needed within a given search window to recenter next search]
	[frame_history: Number of frames to maintain in lane line history and use for averaging]

	Outputs:
	[N/A --- All information (calculated y/x-pixels, polynomial coefficients, turn radius, and center offset) stored in Line() objects. This information is available to the pipeline and used in projecting lane lines, radius, and offset on the image.]

	'''

	# -------- turning radius ---------
	def rad_find(poly2, y):
		'''
		Function:
		Calcualte the point-wise radius for a second degree polynomial function.

		Inputs:
		[poly: Second degree polynomial coefficients]
		[y: Point to calculate radius at]
		'''
		df_dy = 2*poly2[0]*y + poly2[1]
		d2f_df2 = 2*poly2[0]
		r = ((1+df_dy**2)**(3/2))/abs(d2f_df2)
		return r

	Y_tot, X_tot = img_warp.shape
	X_m = int(round(X_tot/2,0))
	Y_win = int(round(Y_tot/win_cnt,0))
	l_ind_a, r_ind_a = [], []

	# define non-zero pixel indices (i.e., valid threshold values)
	img_nz = img_warp.nonzero()
	img_nzy = np.array(img_nz[0])
	img_nzx = np.array(img_nz[1])
	

	
	if line_l.use_poly & line_r.use_poly:
		# use previously calculated polynomial coefficients to find new frame pixels.
		poly_l = line_l.smooth_poly[0]*img_nzy**2 + line_l.smooth_poly[1]*img_nzy + line_l.smooth_poly[2]
		poly_r = line_r.smooth_poly[0]*img_nzy**2 + line_r.smooth_poly[1]*img_nzy + line_r.smooth_poly[2]
		win_xlmin = poly_l - x_mrg
		win_xlmax = poly_l + x_mrg
		win_xrmin = poly_r - x_mrg
		win_xrmax = poly_r + x_mrg
		l_ind_a = ((img_nzx > win_xlmin) & (img_nzx < win_xlmax)).nonzero()[0]
		r_ind_a = ((img_nzx > win_xrmin) & (img_nzx < win_xrmax)).nonzero()[0]
	else:
		# --------- windowed search ----------
		# define initial starting points
		xlb_init = np.argmax(np.sum(img_warp[int(3/4*Y_tot):Y_tot,:X_m], axis=0))
		xrb_init = np.argmax(np.sum(img_warp[int(3/4*Y_tot):Y_tot,X_m:], axis=0))
		xb_off = (X_m - (xrb_init + xlb_init))/2
		# gather all windowed non-zero indices
		xlb_now = xlb_init
		xrb_now = X_m+xrb_init
		for i in range(win_cnt):
			# define windows
			win_ymin = Y_tot - ((i+1) * Y_win)
			win_ymax = Y_tot - (i * Y_win)
			win_xlmin = xlb_now - x_mrg
			win_xlmax = xlb_now + x_mrg
			win_xrmin = xrb_now - x_mrg
			win_xrmax = xrb_now + x_mrg
			# return all non-zero elements within the L/R windows
			l_ind = ((img_nzy >= win_ymin) & (img_nzy < win_ymax) & (img_nzx > win_xlmin) & (img_nzx < win_xlmax)).nonzero()[0]
			r_ind = ((img_nzy >= win_ymin) & (img_nzy < win_ymax) & (img_nzx > win_xrmin) & (img_nzx < win_xrmax)).nonzero()[0]
			l_ind_a.append(l_ind)
			r_ind_a.append(r_ind)
			# re-initialize line center if density is greater than defined minimum
			if len(l_ind) > pxl_cnt_min:
				xlb_now = np.int(np.mean(img_nzx[l_ind]))
			if len(r_ind) > pxl_cnt_min:
				xrb_now = np.int(np.mean(img_nzx[r_ind]))
		# combine all found indices, get pixel coordinates for non-zero values in windows
		l_ind_a = np.concatenate(l_ind_a)
		r_ind_a = np.concatenate(r_ind_a)

	# Valid pixel indices for each lane line, used in polynomial fit.
	xl_pxl = img_nzx[l_ind_a]
	yl_pxl = img_nzy[l_ind_a]
	xr_pxl = img_nzx[r_ind_a]
	yr_pxl = img_nzy[r_ind_a]

	
	# PIXEL SPACE: fit polynomial function to the found projected pixel indices
	ploty_pxl = np.linspace(0, img_warp.shape[0], img_warp.shape[0]-1)
	lpoly_pxl = np.polyfit(yl_pxl, xl_pxl, 2)
	rpoly_pxl = np.polyfit(yr_pxl, xr_pxl, 2)
	lradius_pxl = rad_find(lpoly_pxl, np.max(ploty_pxl))
	rradius_pxl = rad_find(rpoly_pxl, np.max(ploty_pxl))
	l_fitx_pxl = lpoly_pxl[0]*ploty_pxl**2 + lpoly_pxl[1]*ploty_pxl + lpoly_pxl[2]
	r_fitx_pxl = rpoly_pxl[0]*ploty_pxl**2 + rpoly_pxl[1]*ploty_pxl + rpoly_pxl[2]
	offset_pxl = (l_fitx_pxl[-1] + r_fitx_pxl[-1])/2 - X_m


	# IMAGE SPACE: fit polynomial function to the scaled pixel dimensions (meters)
	y_pxl2img = 30/720 # 30 meters per 720 pixels
	x_pxl2img = 3.7/700 #  3.7 meters per 200 pixels
	ploty_img = ploty_pxl * y_pxl2img
	lpoly_img = np.polyfit(yl_pxl*y_pxl2img, xl_pxl*x_pxl2img, 2)
	rpoly_img = np.polyfit(yr_pxl*y_pxl2img, xr_pxl*x_pxl2img, 2)
	lradius_img = rad_find(lpoly_img, np.max(ploty_img))
	rradius_img = rad_find(rpoly_img, np.max(ploty_img))
	offset_img = offset_pxl * x_pxl2img


	l_params = {'pts':l_fitx_pxl, 'poly':lpoly_pxl, 'rad':lradius_img, 'off':offset_img}
	r_params = {'pts':r_fitx_pxl, 'poly':rpoly_pxl, 'rad':rradius_img, 'off':offset_img}


	
	append_lines = True # Not ideal --- GN: Idea is to determine when a given frame has high enough lane line resolution to be included in the history and when to switch from window-search to polynomial-margin. NOTE --- Having 'append_lines' always set to true means that the code below is strictly calculated the moving average (window size = frame_history)
	drop_line = (len(line_l.recent_pts) >= frame_history) & (len(line_r.recent_pts) >= frame_history)
	if append_lines | (len(line_l.recent_pts) == 0) | (len(line_r.recent_pts) == 0):
		line_l.update(l_params, drop_line)
		line_r.update(r_params, drop_line)
		line_l.smooth()
		line_r.smooth()
		if append_lines:
			line_l.use_poly = True
			line_r.use_poly = True
	elif (not append_lines) & (line_l.use_poly | line_r.use_poly):
		line_l.use_poly = False
		line_r.use_poly = False


	return None

# ==============================================================================================================

def pipeline(img_bgr):
	'''
	Function:
	Find the lane lines in an image and project them back onto the original image. Methods include: distortion correction, color/gradient thresholding, perspective transforms, window search with polynomial fit.

	Inputs:
	[img_bgr: Image wtih blue, green, red color channels (order matters)]

	Outputs:
	[result: Input image with projected lane lines and information about the turning radius and center offset]

	'''
	# -=-=-=-=-=-=--= UNDISTORT/THRESHOLD/PERSPECTIVE/LINES =-=-=-=-=-=-=-=-
	img_und_bgr = cv2.undistort(img_bgr, mtx, coeff)
	# -------------------------------
	img_channel = channel_threshold(img_und_bgr, channel='S', threshold=(150, 255))
	img_abs_x = abs_sobel_threshold(img_und_bgr, orient='x', threshold=(50,150))
	img_abs_y = abs_sobel_threshold(img_und_bgr, orient='y', threshold=(50,150))
	img_mag = mag_threshold(img_und_bgr, threshold=(70,140))
	img_dir = dir_threshold(img_und_bgr, sobel_kernel=7, threshold=(0.7, 1.3))
	img_combined = np.zeros_like(img_channel)
	img_combined[(img_channel == 1) | (img_abs_x == 1) | ((img_mag == 1) & (img_dir == 1))] = 1
	# -------------------------------
	img_warp, M, Minv = persepctive(img_combined, edges={'b1':760, 'b2':100, 'h1':60, 'h2':270})
	# -------------------------------
	line_find(img_warp, line_l, line_r, win_cnt=20, x_mrg=100, pxl_cnt_min=50, frame_history=5)	


	# =-=-=-=-=-=-=-=-=-=- PROJECTION =-=-=-=-=-=-=-=-=-=-=-
	img_warp_z = np.zeros_like(img_warp).astype(np.uint8) # Create blank three channel image array
	img_warp_z3 = np.dstack((img_warp_z, img_warp_z, img_warp_z))
	l_pts = np.array([np.transpose(np.vstack([line_l.smooth_pts, np.linspace(0, len(line_l.smooth_pts), len(line_l.smooth_pts))]))]) # Concatenate x/y coordinates for cv2.fillPoly()
	r_pts = np.array([np.flipud(np.transpose(np.vstack([line_r.smooth_pts, np.linspace(0, len(line_r.smooth_pts), len(line_r.smooth_pts))])))])
	pts = np.hstack((l_pts, r_pts)) #Stack L/R lines for cv2.fillPoly()
	cv2.fillPoly(img_warp_z3, np.int_([pts]), (0,255, 0))
	img_warp_lines = cv2.warpPerspective(img_warp_z3, Minv, (img_bgr.shape[1], img_bgr.shape[0])) # Warp the blank back to original image space using inverse perspective matrix (Minv)
	img_text = 'Turn Radius (m): {:01.0f} | Center Offset (m): {:02.2f}'.format(line_l.smooth_rad, line_l.smooth_off)
	result = cv2.putText(cv2.addWeighted(img_bgr, 1, img_warp_lines, 0.3, 0), img_text, (50,50), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(50, 200, 50), thickness=2) # Merge images with transparency
	
	'''
	# -=-=-=-=-=-=-=-=-=-=-=-=-=- VISUALIZATION =-=-=-=-=-=-=-=-=-=-=-=-
	f1, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9,8))
	f1.tight_layout()
	ax11.imshow(cv2.cvtColor(img_und_bgr, cv2.COLOR_BGR2RGB))
	ax11.set_title('Undistorted Image', fontsize=10)
	ax12.imshow(img_combined, cmap='gray')
	ax12.set_title('Threshold - Combined', fontsize=10)
	ax21.imshow(img_warp, cmap='gray')
	ax21.set_title('Perspective Transform', fontsize=10)
	ax22.imshow(img_warp, cmap='gray')
	ploty = np.linspace(0, img_warp.shape[0], img_warp.shape[0]-1)
	ax22.plot(line_l.smooth_pts, ploty, color='red', linewidth=3)
	ax22.plot(line_r.smooth_pts, ploty, color='blue', linewidth=3)
	ax22.set_title('Lane Line Detection', fontsize=10)
	ax31.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
	ax31.set_title('Projected Lane Lines', fontsize=10)
	plt.xlim(0,img_bgr.shape[1])
	plt.ylim(img_bgr.shape[0],0)
	plt.show(f1)


	f2, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9,8))
	f2.tight_layout()
	ax11.imshow(img_combined, cmap='gray')
	ax11.set_title('Threshold - Combined', fontsize=10)
	ax12.imshow(img_abs_x, cmap='gray')
	ax12.set_title('X_gradient', fontsize=10)
	ax21.imshow(img_abs_y, cmap='gray')
	ax21.set_title('Y_gradient', fontsize=10)
	ax22.imshow(img_mag, cmap='gray')
	ax22.set_title('Magnitude', fontsize=10)
	ax31.imshow(img_dir, cmap='gray')
	ax31.set_title('Direction', fontsize=10)
	ax32.imshow(img_channel, cmap='gray')
	ax32.set_title('Color Channel Threshold', fontsize=10)
	plt.show(f2)
	'''


	return result

# ==============================================================================================================

class LineParams():
	def __init__(self):
		self.use_poly = False

		self.recent_pts = []
		self.smooth_pts = []

		self.recent_poly = []
		self.smooth_poly = []
		
		self.recent_rad = []
		self.smooth_rad = 0

		self.recent_off = []
		self.smooth_off = 0

	def update(self, params, drop=False):
		self.recent_pts.append(params['pts'])
		self.recent_poly.append(params['poly'])
		self.recent_rad.append(params['rad'])
		self.recent_off.append(params['off'])
		if drop:
			self.recent_pts = self.recent_pts[1:]
			self.recent_poly = self.recent_poly[1:]

	def smooth(self):
		self.smooth_pts = np.average(self.recent_pts, axis=0)
		self.smooth_poly = np.average(self.recent_poly, axis=0)
		self.smooth_rad = np.average(self.recent_rad, axis=0)
		self.smooth_off = np.average(self.recent_off, axis=0)
		self.recent_rad = []
		self.recent_off = []

line_l = LineParams()
line_r = LineParams()
mtx, coeff = camera_cal(glob.glob('/users/graham.nekut/dev/udacity/carnd/p4/Carnd-Advanced-Lane-Lines/camera_cal/calibration*.jpg'), (9,6))

video_in = VideoFileClip('/users/graham.nekut/dev/udacity/carnd/p4/Carnd-Advanced-Lane-Lines/challenge_video.mp4')
video_clip = video_in.fl_image(pipeline)
video_clip.write_videofile('/users/graham.nekut/dev/udacity/carnd/p4/c_video_out.mp4', audio=False)

'''
x = pipeline(cv2.imread('/users/graham.nekut/dev/udacity/carnd/p4/Carnd-Advanced-Lane-Lines/test_images/test2.jpg'))
'''
