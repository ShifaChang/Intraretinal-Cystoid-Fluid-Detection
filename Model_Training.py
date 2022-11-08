import datetime

mylog = "./logs"

#ModelCheckpoint
callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        #tf.keras.callbacks.TensorBoard(log_dir=mylog + datetime.datetime.now().strftime("%d %m %Y c")),
        #tf.keras.callbacks.ModelCheckpoint('../input/intraretinal-cystoid-fluid/model_eye_diseases7_denoised.h5', save_best_only=True, save_weights_only=True, verbose=2)
        ]

results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=8, epochs=50, callbacks=callbacks)