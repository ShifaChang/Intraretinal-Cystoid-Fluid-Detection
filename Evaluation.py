model = create_model()

# Load weights
model.load_weights('../input/intraretinal-cystoid-fluid/model_eye_diseases7_denoised.h5')

loss,acc,auc,dc,rec,prec = model.evaluate(X_train, Y_train, verbose=2)

print("Restored model, accuracy:                     {:5.2f}%".format(100*acc))
print("Restored model, Loss:                         {:5.2f}%".format(100*loss))
print("Restored model, Area Under the Curve (AUC):   {:5.2f}%".format(100*auc))
print("Restored model, Recall:                       {:5.2f}%".format(100*rec))
print("Restored model, Precesion:                    {:5.2f}%".format(100*prec))