# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

plt.rcParams["figure.figsize"] = (15.0, 12.0)  # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'


#%%
with open("cache/training/neg_samples.pkl", "rb") as f:
    neg_samples = pickle.load(f)
with open("cache/training/waldo.pkl", "rb") as f:
    pos_waldo = pickle.load(f)
with open("cache/training/wenda.pkl", "rb") as f:
    pos_wenda = pickle.load(f)
with open("cache/training/wizard.pkl", "rb") as f:
    pos_wizard = pickle.load(f)

#%% [markdown]
# ## Compute HOG feature descriptor for all samples

#%%
from skimage.feature import hog

# DESCRIPTOR CONFIG
ORIENTATIONS = 8
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

# viualize the descriptor once for each class
for i, s in enumerate([pos_waldo[0], pos_wenda[0], pos_wizard[0], neg_samples[0]]):
    feature, hog_image = hog(
        s,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        visualize=True,
        multichannel=True,
    )
    print(
        "Sample image shape {}, feature vector shape {}".format(s.shape, feature.shape)
    )
    plt.subplot(4, 2, 2 * i + 1)
    plt.imshow(s)
    plt.subplot(4, 2, 2 * i + 2)
    plt.imshow(hog_image)

plt.show()


#%%
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from joblib import Parallel, delayed, cpu_count


LABELS = {"waldo": 0, "wenda": 1, "wizard": 2, "negative": 3}


def hog_descriptor(samples):
    f_vects = Parallel(n_jobs=cpu_count())(
        delayed(hog)(
            s,
            orientations=ORIENTATIONS,
            pixels_per_cell=PIXELS_PER_CELL,
            cells_per_block=CELLS_PER_BLOCK,
            visualize=False,
            multichannel=True,
        )
        for s in samples
    )
    return np.array(f_vects)


# transform samples into feature space
pf_wa = hog_descriptor(pos_waldo)
pf_we = hog_descriptor(pos_wenda)
pf_wi = hog_descriptor(pos_wizard)
nf_s = hog_descriptor(neg_samples)

X = np.concatenate((pf_wa, pf_we, pf_wi, nf_s), axis=0)
Y = np.concatenate(
    (
        np.full(pf_wa.shape[0], LABELS["waldo"]),
        np.full(pf_we.shape[0], LABELS["wenda"]),
        np.full(pf_wi.shape[0], LABELS["wizard"]),
        np.full(nf_s.shape[0], LABELS["negative"]),
    ),
    axis=0,
)
label_bin = preprocessing.LabelBinarizer()
label_bin.fit(list(LABELS.values()))
Y = label_bin.transform(Y)

x_train, y_train = X, Y
print("training data shape: examples: {}, labels: {}".format(x_train.shape, y_train.shape))

#%% [markdown]
# ## Train Support Vector Machine Classifier (SVC)

#%%
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import cross_val_score

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(x_train.shape, y_train.shape)

clf = OneVsRestClassifier(svm.LinearSVC(), n_jobs=-1)
clf.fit(x_train, y_train)

# save model to disk
with open('cache/trained_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

#%% [markdown]
# ## Evaluate Classifier

#%%
# load testing data
with open("cache/testing/waldo.pkl", "rb") as f:
    test_waldo = pickle.load(f)
with open("cache/testing/wenda.pkl", "rb") as f:
    test_wenda = pickle.load(f)
with open("cache/testing/wizard.pkl", "rb") as f:
    test_wizard = pickle.load(f)
with open("cache/testing/neg_samples.pkl", "rb") as f:
    test_neg_samples = pickle.load(f)

# transform samples into feature space
test_waldo = hog_descriptor(test_waldo)
test_wenda = hog_descriptor(test_wenda)
test_wizard = hog_descriptor(test_wizard)
test_neg_samples = hog_descriptor(test_neg_samples)

x_test = np.concatenate((test_waldo, test_wenda, test_wizard, test_neg_samples), axis=0)
y_test = np.concatenate(
    (
        np.full(test_waldo.shape[0], LABELS["waldo"]),
        np.full(test_wenda.shape[0], LABELS["wenda"]),
        np.full(test_wizard.shape[0], LABELS["wizard"]),
        np.full(test_neg_samples.shape[0], LABELS["negative"]),
    ),
    axis=0,
)

y_test = label_bin.transform(y_test)


#%%
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

y_score = clf.decision_function(x_test)
precision = dict()
recall = dict()

average_precision = dict()
for i in range(len(LABELS)):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
precision["micro"], recall["micro"], _ = precision_recall_curve(
    y_test.ravel(), y_score.ravel()
)
average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
print(
    "Average precision score, micro-averaged over all classes: {0:0.2f}".format(
        average_precision["micro"]
    )
)

plt.subplot()
plt.step(recall["micro"], precision["micro"], color="b", alpha=0.2, where="post")
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color="b", step="post")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    "Average precision score, micro-averaged over all classes: AP={0:0.2f}".format(
        average_precision["micro"]
    )
)
plt.show()


#%%

ax = plt.subplot(2,2,1)
plt.title('Waldo test result distribution')
plt.hist(label_bin.inverse_transform(clf.predict(test_waldo)))
# ax.set_xticklabels(LABELS ,rotation=45, rotation_mode="anchor", ha="right")

ax = plt.subplot(2,2,2)
plt.title('Wenda test result distribution')
plt.hist(label_bin.inverse_transform(clf.predict(test_wenda)))
# ax.set_xticklabels(LABELS ,rotation=45, rotation_mode="anchor", ha="right")

ax = plt.subplot(2,2,3)
plt.title('Wizard test result distribution')
plt.hist(label_bin.inverse_transform(clf.predict(test_wizard)))
# ax.set_xticklabels(LABELS ,rotation=45, rotation_mode="anchor", ha="right")

ax = plt.subplot(2,2,4)
plt.title('Negative Samples test result distribution')
plt.hist(label_bin.inverse_transform(clf.predict(test_neg_samples)))
# ax.set_xticklabels(LABELS ,rotation=45, rotation_mode="anchor", ha="right")
plt.show()


#%%

