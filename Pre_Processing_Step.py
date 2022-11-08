IMG_WIDTH = 320
IMG_HEIGHT = 320
IMG_CHANNELS = 3


TRAIN_PATH = '/kaggle/input/intraretinal-cystoid-fluid/2021-training-data-ZA/2021-training-data-ZA/'

train_ids = next(os.walk(TRAIN_PATH))[1]
print('length of train IDs: ', len(train_ids))


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1),  dtype='bool')

print('X_train shape ', X_train.shape)
#print(X_train.dtype)
#print(X_train)

print('Y_train shape ', Y_train.shape)
#print(Y_train.dtype)
#print(Y_train)
alpha = 1.4  # Contrast control (1.0-3.0)
beta = 0  # Brightness control (0-100)
psf = np.ones((5, 5)) / 25  # psf = point spread function for richardson_lucy algorithm
'''
#FOR TRAINING IMAGES
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    #print(path)
    img = img_as_float(cv2.imread(path + '/images/' + id_ + '.jpeg', 0))
    #img = gray2rgb(img)
    #img = img[:,:,:IMG_CHANNELS]
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    #print(img.shape) #320 320 3

    min_img = ndimage.filters.minimum_filter(img, size=3 , output=None, mode='wrap', cval=0.0, origin=0) #morphological filter

    BM3D_denoised_image = bm3d.bm3d(min_img, sigma_psd=0.08, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING, profile='high') #denoising

    # Restore Image using Richardson-Lucy algorithm
    deconvolved_RL = restoration.richardson_lucy(BM3D_denoised_image, psf, iterations=5) #regularization= iteration
    deconvolved_RL = (deconvolved_RL*255).astype('uint8')

    adjusted = cv2.convertScaleAbs((deconvolved_RL), alpha=alpha, beta=beta) #contrast

    adj3d = gray2rgb(adjusted)
    X_train[n] = adj3d


    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype='bool')
    #print(mask.shape)    
    #now mask folder, read all mask images
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = cv2.imread(path + '/masks/' + mask_file)  
        #print(mask_.shape) #512 512 3
        mask_ = cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
        #print('okkkkkk')
        #print(mask_.shape) #320 320 3
        mask_ = (mask_)[:,:,:1]
        #print(mask_.shape) # 320 320 1
        #1=fluid and 0=black
        mask = np.maximum(mask, mask_) #it will take the max. pixel value(e.g. 1) at each pixel location from those masks.
        #print(mask.shape)

    Y_train[n] = mask
'''

print('done')