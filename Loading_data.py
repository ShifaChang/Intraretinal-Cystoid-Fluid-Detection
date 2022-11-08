# loading preprocessed data

import pickle
'''
pickle_out = open("/kaggle/input/intraretinal-cystoid-fluid/X_train_1006.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("/kaggle/input/intraretinal-cystoid-fluid/Y_train_1006.pickle","wb")
pickle.dump(Y_train, pickle_out)
pickle_out.close()
'''
#We can always load it in to our current script, or a totally new one by doing:

pickle_in = open("/kaggle/input/intraretinal-cystoid-fluid/X_train_1006.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("/kaggle/input/intraretinal-cystoid-fluid/Y_train_1006.pickle","rb")
Y_train = pickle.load(pickle_in)
plt.imshow(X_train[9])
plt.show()
plt.imshow(np.squeeze(Y_train[9]))