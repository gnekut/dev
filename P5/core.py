import cv2 # Image processing
import glob # File name extraction
import time # Training time determination
import pickle # Saving and loading classifier and scaler
import numpy as np # Matrix manipulations
import matplotlib.pyplot as plt # Image plotting
from skimage.feature import hog # Histogram of Oriented Gradients feature extraction
from sklearn.preprocessing import StandardScaler # Normalization scaler of training/pipeline image features
from sklearn.model_selection import train_test_split # Randomized Train/Test data splitting
from scipy.ndimage.measurements import label # Determination of distinct heatmap object
from sklearn.utils import shuffle # Shuffle directory file names before subsampling to eliminate time-series dependence
from sklearn.svm import SVC # Linear support vector machine model
from moviepy.editor import VideoFileClip # Video processing

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def input_validator(options,msg,info=''):
	'''
	Function: (Note, pulled from another project). Inpurt validation to ensure only valid options are allowed.

	Inputs: 
	[options: list/tuple of valid values]
	[msg: display message to user]
	[infor: addition display message to show user outside of input prompt message]

	Outputs: User selection
	'''
	if info != '':
		print(info)
	while 'sel' not in locals():
		sel = input('%s'%msg).lower()
		if sel not in options:
			print('\nInvalid Entry [%s]. Select from [%s]'%(sel,options))
	return sel


def extract_files(roots, subset=1):
	'''
	Function: Extract all PNG-picture files within a given set of directories and return only a subset from each of them.
	
	Inputs:
	[roots: Root level directories to search]
	[subset: subset of found files to return from each directory]

	Outputs: List of found files
	'''
	trn_files = []
	for root in roots:
		img_files = glob.glob(root+'/*.png')
		trn_files.extend(shuffle(img_files)[:(len(img_files)//subset)])
	return trn_files


def channel_selection(img, cs=None, cc=None, valid_cs=['RGB', 'HLS', 'HSV', 'GRAY', 'YUV', 'LUV', 'YCrCb']):
	'''
	Function: Conver image to a different channel and select an image channel (opt)

	Inputs:
	[img: Image input]
	[cs: Color space to convert to]
	[cc: Color channel selection [0-2]
	[valid_cs: List of valid color spaces that an image can be converted to]

	Outputs: Converted image
	'''
	if cs in valid_cs:
		img = cv2.cvtColor(img, eval('cv2.COLOR_BGR2%s'%(cs)))
	if cc != None:
		img = img[:,:,cc]
	return img

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def pxl_extract(img, cs=None, cc=None, size=(32, 32)):
	'''
	Function: Spatial feature extraction ('pxl': pixel)

	Inputs:
	[img: Image input]
	[cs: see channel selection]
	[cc: see channel selection]
	[size: pixelated resolution of resulting image]

	Outputs: Flattened spatial feature set of length Y-dimension x X-dimension x Channel Cnt
	'''
	img = channel_selection(img, cs, cc)
	return cv2.resize(img, size).ravel()


def chnl_extract(img, cs=None, cc=None, nbins=32, bins_range=(0, 256)):
	'''
	Function: Extract color channel histogram features. Each channel is broken down into nbins-channels.

	Inputs:
	[cs: see channel selection]
	[cc: see channel selection]
	[nbins: Resolution of a channels color space (i.e., number of histogram bins). Each bin is treated a feature element.]
	[bin_range: Bounds of the histogram binning.]

	Outputs: Flattened color channel extraction feature set of length (Channel Cnt x Number of histogram bins)
	'''
	img = channel_selection(img, cs, cc)
	chnl_hsts = []
	for i in range(img.shape[2]):
		chnl_hsts.append(np.histogram(img[:,:,0], bins=nbins, range=bins_range)[0])
	return np.concatenate(chnl_hsts)


def hog_extract(img, cs=None, orient=9, ppc=8, cpb=2, feature_vector=True):
	'''
	Function: Histogram of Oriented Gradient (HOG) analyzes images in accordance with each pixels local gradients, collecting all of the gradients in a predifined spatial region. Each spatial region is defined by a certain number of N-pixels per cell (ppc) where the gradient direction of each pixel fall into one of O-bins (think of a histogram based upon direction where the O-number of bins is defined by 'orientations' (orient)). Now each cell of ppc**2 pixels can be expressed as a collection of orientations containing some number of pixels (0 to ppc**2). Cells are structured into a set of blocks with M cells per block (cpb). The process steps through an image 1-cell at a time, analyzing each as a new block, therefore producing more features than without overlap. In the this project, images were analyzed at a size of 64 x 64 pixels ----- Therefore, O-orientations, M-cells per block, (Size - N-pixels per cell)/N-pixels per cell ----- O*M*M*(N-1)*(N-1)

	Inputs:
	[img: Image input]
	[orients: Number of pixel gradient orientations to be applied per cell (i.e., gradient orientation histogram)]
	[ppc: Pixels per cell]
	[cpb: Cells per block]
	[feature_vector: (True: Perform ravel() on feature set before returning), (False: Return in matrix form)]

	Outputs: Extracted HOG features in single vectorized feature or in matrix for for spatial subsampling.
	'''
	features, images = [], []
	for i in range(img.shape[-1]):
		features.append(hog(channel_selection(img, cs, i), orientations=orient, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb), block_norm='L1', visualise=False, feature_vector=feature_vector, transform_sqrt=True))
	
	if feature_vector:
		return np.concatenate(features)
	else:
		return features


def extract_features(image, feature_vector=True):
	'''
	Function: Utility function to aggregate all features for a given image.

	Inputs:
	[img: Image input]
	[feature_vector: See hog_extract(), if True individual feature sets will be returned.]

	Outputs: Aggregated feature set for an image (single vector if feature_vector=True, individual components if False)
	'''
	pxls = pxl_extract(image, ftr_param.pxls['cs'], ftr_param.pxls['cc'], ftr_param.pxls['size'])
	chnls = chnl_extract(image, ftr_param.chnls['cs'], ftr_param.chnls['cc'], ftr_param.chnls['n_bins'], ftr_param.chnls['bins_range'])
	hogs = hog_extract(image, ftr_param.hogs['cs'], ftr_param.hogs['n_orient'], ftr_param.hogs['ppc'], ftr_param.hogs['cpb'], feature_vector=feature_vector)
	if feature_vector:
		features = np.concatenate((pxls, chnls, hogs))
		return features
	else:
		return pxls, chnls, hogs


def compile_features(files):
	'''
	Function: Utility function to compile all features for a set of give files

	Inputs:
	[files: List of image files to extract features from]

	Outputs: Two dimensional array (Size: Feature length x Sample size)
	'''
	features_list = []
	for f in files:
		image = cv2.imread(f)
		features_list.append(extract_features(image))
	return features_list

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def train_classifier():
	'''
	Function: Extracts training/test data sets and implements a Linear Support Vector Machine (L-SVM), saving the resulting model for later use.

	Inputs: N/A

	Outputs:
	[sclr: Saved feature normalization scalar, loaded in image/video processing]
	[clfr: Saved Linear Support Vector Machine (L-SVM), loaded in image/video processing]
	'''
	print('\n\n\nExtracting Features...')
	c_features = compile_features(shuffle(extract_files(glob.glob('/Users/graham.nekut/dev/udacity/carnd/p5/vehicles/*'))))
	nc_features = compile_features(shuffle(extract_files(glob.glob('/Users/graham.nekut/dev/udacity/carnd/p5/non-vehicles/*'))))

	features_a = np.concatenate((c_features, nc_features)).astype(np.float64)
	labels_a = np.concatenate((np.ones(len(c_features)), np.zeros(len(nc_features))))
	sclr = StandardScaler().fit(features_a)
	features_norm_a = sclr.transform(features_a)

	ftrs_train, ftrs_test, lbls_train, lbls_test = train_test_split(features_norm_a, labels_a, test_size=0.2, random_state=np.random.randint(0, 100))
	print('Train/Test Split (samples,features):  ', ftrs_train.shape, ' / ', ftrs_test.shape)


	print('\nTraining Classifier.....')
	clfr = SVC(kernel='linear', C=1)
	t1=time.time()
	clfr.fit(ftrs_train, lbls_train)
	t2=time.time()
	print('Training Time (sec): ' , t2-t1 )
	print('Test Accuracy of SVC: ', clfr.score(ftrs_test, lbls_test))
	
	with open('clfr.pickle', 'wb') as file:
		pickle.dump(clfr, file)
	with open('sclr.pickle', 'wb') as file:
		pickle.dump(sclr, file)
	print('[Model Saved.]\n\n\n')

	return None

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def pipeline(img_in, vis=False, hog_subsample=False):
	'''
	Function: Track trained objects in the supplied image(s).

	Inputs:
	[img_in: Video frame or supplied image.]
	[vis: (boolean) Visualize framed image.]
	[subsample: (boolean) Use subsampling technique to extract HOG features (Note: Implementation appears correct and there is a speed improvement (20-30% faster); however, results never appeared to be as good as extracting individual elements. After asking other people how they implemented it, it seems mine is identical.)]

	Outputs: Image(s) with projected vehicle frames.
	'''

	# -----------------------------------------------------------------
	# Define image sclaes, image trimming and iteration parameters
	scales = [1.25, 1.5]
	y_trims = [(320, 220), (320, 180)]
	x_trims = [(80,80), (50,50)]
	ppf = 64
	pps = 32
	ppc = ftr_param.hogs['ppc']
	cpb = ftr_param.hogs['cpb']
	cpf = ppf // ppc - 1
	cps = pps // ppc - 1
	current_frames = []
	for s, scale in enumerate(scales):
		

		# -----------------------------------------------------------------
		# Trim and scale image
		y_min_trim = (img_in.shape[0] - y_trims[s][0])
		y_max_trim = (img_in.shape[0] - y_trims[s][1])
		x_min_trim = (0 + x_trims[s][0])
		x_max_trim = (img_in.shape[1] - x_trims[s][1])
		img_scaled = img_in[y_min_trim:y_max_trim, x_min_trim:x_max_trim,:]
		if scale != 1:
			img_scaled = cv2.resize(img_scaled, (int(img_scaled.shape[1]//scale),int(img_scaled.shape[0]//scale)))
		nsteps_x = (img_scaled.shape[1] - ppf) // pps + 1
		nsteps_y = (img_scaled.shape[0] - ppf) // pps + 1

		# HOG subsampling
		if hog_subsample:
			hog_channel_features = hog_extract(img_scaled, ftr_param.hogs['cs'], ftr_param.hogs['n_orient'], ftr_param.hogs['ppc'], ftr_param.hogs['cpb'], feature_vector=False)


		# -----------------------------------------------------------------
		# Window search iterations
		for j in range(nsteps_y):
			for i in range(nsteps_x):
				# Define and select current sample
				xp_min = i * pps
				yp_min = j * pps
				xc_min = i * cps
				yc_min = j * cps
				img_sample = cv2.resize(img_scaled[yp_min:yp_min+ppf, xp_min:xp_min+ppf], (64,64))

				# Image sample feature extraction
				if hog_subsample:
					pxls = pxl_extract(img_sample, ftr_param.pxls['cs'], ftr_param.pxls['cc'], ftr_param.pxls['size'])
					chnls = chnl_extract(img_sample, ftr_param.chnls['cs'], ftr_param.chnls['cc'], ftr_param.chnls['n_bins'], ftr_param.chnls['bins_range'])
					hogs = []
					for k in range(len(hog_channel_features)):
						hogs.append(hog_channel_features[k][yc_min:yc_min+cpf,xc_min:xc_min+cpf])
					hogs = np.concatenate(hogs).ravel()
					all_features = np.concatenate((pxls, chnls, hogs)).astype(np.float64)
				else:
					all_features = extract_features(img_sample).astype(np.float64)
				
				# Normalize and classify sample, decide if automobile is detected
				if clfr.decision_function(sclr.transform(all_features)) > 0.5:
					x_min_scale = int(xp_min * scale)
					y_min_scale = int(yp_min * scale)
					ppf_scale = int(ppf * scale)
					current_frames.append(((x_min_scale+x_min_trim, y_min_scale+y_min_trim), (x_min_scale+x_min_trim+ppf_scale, y_min_scale+y_min_trim+ppf_scale)))
	
	# -----------------------------------------------------------------
	# Add current frames to buffer, adjust buffer, and generate found automobile frames
	frames.buffer(img_in, current_frames)
	
	return frames.draw(img_in, frames.valid_frames, ((x_min_trim,y_min_trim),(x_max_trim,y_max_trim)))


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class FeatureParameters():
	'''
	Properties:
	[chnls: (cs: Color space coversion), (cc: Color channel selection), (n_bins: Numer of bins used in histogram), (bins_range: Range of values expected by histogram)]
	[pxls: (cs: ''), (cc: ''), (size: Pixelation resolution)]
	[hogs: (cs: ''), (cc: ''), (n_orients: Number of gradient orientations per cell), (ppc: Pixel per cell in both X and Y dimensions (i.e. total count is ppc**2), (cpb: Cells per block in both X and Y dimensions (i.e. each step of the feature extraction is cpb**2 * n_orients)))]

	Methods: N/A
	'''
	def __init__(self):
		self.chnls = {'cs': 'HLS', 'cc': None, 'n_bins': 32, 'bins_range': (0,256)}
		self.pxls = {'cs': None, 'cc': None, 'size': (32,32)}
		self.hogs = {'cs': 'YUV', 'cc': None, 'n_orient': 9, 'ppc': 8, 'cpb': 2}

class Frames():
	'''
	Properties:
	[buffered_heatmaps: Heatmap buffer of past N image frames (N = buffer(max_frames=N))]
	[valid_heatmap: Current frame's heatmap aggregation]
	[valid_labels: Current frame's determined label objects]
	[valid_frames: Current frame's found object(s) coordinates]

	Methods:
	[build_heatmap: Build heatmap based upon the array of found object coordinates]
	[label_frames: Identify found object(s) based upon previously buffered heatmaps]
	[buffer: Construct buffered heatmap object(s), passing to labeling method]
	[draw: Draw rectangles on a given image, returning the new image to calling space (Note: vis=True will call drawn image to display)]


	'''
	def __init__(self):
		self.max_frames = 10
		self.heatmap_threshold = 10
		self.buffered_heatmaps = []
		self.valid_heatmap = []
		self.valid_labels = []
		self.valid_frames = []

	def build_heatmap(self, img, frames):
		heatmap = np.zeros_like(img[:,:,0])
		for f in frames:
			heatmap[f[0][1]:f[1][1], f[0][0]:f[1][0]] += 1
		return heatmap

	def label_frames(self, img):
		
		hmap_agg = np.zeros_like(img[:,:,0])
		for hmap in self.buffered_heatmaps:
			hmap_agg += hmap
		hmap_agg[hmap_agg < self.heatmap_threshold] = 0
		self.valid_heatmap = hmap_agg


		self.valid_labels = label(hmap_agg)
		labeled_frames = []
		for i in range(self.valid_labels[1]):
			nz_i = (self.valid_labels[0] == i+1).nonzero()
			labeled_frames.append([[np.min(nz_i[1]), np.min(nz_i[0])], [np.max(nz_i[1]), np.max(nz_i[0])]])
		self.valid_frames = labeled_frames
		return None


	def buffer(self, img, input_frames):
		if len(input_frames) > 0:
			self.buffered_heatmaps.append(self.build_heatmap(img, input_frames))
		if len(self.buffered_heatmaps) > self.max_frames:
				self.buffered_heatmaps = self.buffered_heatmaps[1:]
		self.label_frames(img)
		return None


	def draw(self, img, frames, search_bounds=None, vis=False):
		img_out = np.copy(img)
		for f in frames:
			cv2.rectangle(img_out, tuple(f[0]), tuple(f[1]), color=(0,255,0), thickness=2)
		if search_bounds != None:
			cv2.rectangle(img_out, (search_bounds[0][0],search_bounds[0][1]), (search_bounds[1][0],search_bounds[1][1]), color=(255,0,255), thickness=2)
		return img_out


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

run = input_validator(('a','c','f','v'),'[A:All, C:Classifier, F:Frames, V:Video]: ')
# ------ Initialize Classes ------
frames = Frames()
ftr_param = FeatureParameters()

# ------ Train Classifier --------
if run in ['c','a']:
	train_classifier()

# ------ Run Detection ---------
if run in ['f','a','v']:
	with open('clfr.pickle', 'rb') as file:
		clfr = pickle.load(file)
	with open('sclr.pickle', 'rb') as file:
		sclr = pickle.load(file)

	# ------ Run Detection - Sample Images -------
	# Check sample images for detection
	if run == 'f':
		frames.heatmap_threshold = 2
		for imgs in glob.glob('/Users/graham.nekut/dev/udacity/carnd/p5/CarND-Car-Detection/test_images/*.jpg'):
			img = cv2.imread(imgs)
			img_out = pipeline(img)

			f1, ((ax11, ax12, ax13)) = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(5,8))
			f1.tight_layout()
			ax11.imshow(frames.valid_heatmap)
			ax11.set_title('Image Heatmap', fontsize=10)
			ax12.imshow(frames.valid_labels[0], cmap='gray')
			ax12.set_title('Labled Object(s)', fontsize=10)
			ax13.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
			ax13.set_title('Detected Object(s)', fontsize=10)
			plt.show(f1)
			frames = Frames()


	# ------- Run Detection - Project Video ---------
	# Build video output
	duration = (0,50) # Video duration bounds (seconds)
	fps = 20 # Frame resolutions (frames per second)
	if run in ['a','v']:
		video_in = VideoFileClip('/users/graham.nekut/dev/udacity/carnd/p5/CarND-Car-Detection/project_video.mp4').subclip(duration[0],duration[1]).set_fps(fps)
		video_clip = video_in.fl_image(pipeline)
		video_clip.write_videofile('/users/graham.nekut/dev/udacity/carnd/p5/video_out2.mp4', audio=False)




