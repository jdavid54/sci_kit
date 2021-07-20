import cython

from cython.parallel import prange

# print(dir(cython))
'''
['ArrayType', 'CythonDotParallel', 'CythonMetaType', 'CythonType', 'CythonTypeObject',
'NULL', 'PointerType', 'Py_UCS4', 'Py_UNICODE', 'Py_ssize_t', 'Py_tss_t', 'StructType',
'UnionType', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__',
'__package__', '__spec__', '__version__', 'absolute_import', 'address',
'always_allows_keywords', 'array', 'basestring', 'binding', 'bint', 'boundscheck',
'cast', 'ccall', 'cclass', 'cdiv', 'cdivision', 'cdivision_warnings', 'cfunc', 'char',
'cmod', 'compile', 'compiled', 'complex', 'complex_types', 'declare', 'double',
'doublecomplex', 'embedsignature', 'exceptval', 'final', 'float', 'float_types',
'floatcomplex', 'floating', 'freelist', 'fused_type', 'gil', 'gs', 'i', 'index_type',
'infer_types', 'initializedcheck', 'inline', 'int', 'int_types', 'integral', 'internal',
'linetrace', 'load_ipython_extension', 'locals', 'long', 'longdouble', 'longdoublecomplex',
'longlong', 'name', 'no_gc', 'no_gc_clear', 'nogil', 'nonecheck', 'numeric', 'optimization',
'other_types', 'overflowcheck', 'p_Py_UCS4', 'p_Py_UNICODE', 'p_Py_ssize_t', 'p_Py_tss_t',
'p_bint', 'p_char', 'p_complex', 'p_double', 'p_doublecomplex', 'p_float', 'p_floatcomplex',
'p_int', 'p_long', 'p_longdouble', 'p_longdoublecomplex', 'p_longlong', 'p_short',
'p_size_t', 'p_void', 'pointer', 'pp_Py_UCS4', 'pp_Py_UNICODE', 'pp_Py_ssize_t',
'pp_Py_tss_t', 'pp_bint', 'pp_char', 'pp_complex', 'pp_double', 'pp_doublecomplex',
'pp_float', 'pp_floatcomplex', 'pp_int', 'pp_long', 'pp_longdouble', 'pp_longdoublecomplex',
'pp_longlong', 'pp_short', 'pp_size_t', 'pp_void', 'ppp_Py_UCS4', 'ppp_Py_UNICODE',
'ppp_Py_ssize_t', 'ppp_Py_tss_t', 'ppp_bint', 'ppp_char', 'ppp_complex', 'ppp_double',
'ppp_doublecomplex', 'ppp_float', 'ppp_floatcomplex', 'ppp_int', 'ppp_long',
'ppp_longdouble', 'ppp_longdoublecomplex', 'ppp_longlong', 'ppp_short', 'ppp_size_t',
'ppp_void', 'profile', 'py_complex', 'py_float', 'py_int', 'py_long', 'reprname',
'returns', 'schar', 'short', 'sint', 'size_t', 'sizeof', 'slong', 'slonglong',
'sshort', 'struct', 't', 'test_assert_path_exists', 'test_fail_if_path_exists',
'to_repr', 'type_ordering', 'type_version_tag', 'typedef', 'typeof', 'uchar',
'uint', 'ulong', 'ulonglong', 'unicode', 'union', 'unraisable_tracebacks', 'ushort',
'void', 'wraparound']
'''

# cimport problem
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.inspection import plot_partial_dependence
# from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.inspection import plot_partial_dependence


# cannot import name
# from sklearn.model_selection import HalvingRandomSearchCV

# clf = RandomForestClassifier(random_state=0)
# X = [[ 1,  2,  3],  # 2 samples, 3 features
#      [11, 12, 13]]
# y = [0, 1]  # classes of each sample
# clf.fit(X, y)

from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]
# scale data according to computed scaling values
y = StandardScaler().fit(X).transform(X)
print(X,y)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor

from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
import scipy

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import completeness_score

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

from scipy.stats import randint

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
'''

clf = classifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
'''
rng = np.random.RandomState(0)
X, y = make_blobs(random_state=rng)
X = scipy.sparse.csr_matrix(X)
X_train, X_test, _, y_test = train_test_split(X, y, random_state=rng)
kmeans = KMeans(algorithm='elkan').fit(X_train)
print(completeness_score(kmeans.predict(X_test), y_test))

n_samples, n_features = 1000, 20
rng = np.random.RandomState(0)
X, y = make_regression(n_samples, n_features, random_state=rng)
sample_weight = rng.rand(n_samples)
X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
    X, y, sample_weight, random_state=rng)
reg = Lasso()
reg.fit(X_train, y_train, sample_weight=sw_train)
print(reg.score(X_test, y_test, sw_test))

rng = np.random.RandomState(42)
iris = datasets.load_iris()
random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
iris.target[random_unlabeled_points] = -1
svc = SVC(probability=True, gamma="auto")
self_training_model = SelfTrainingClassifier(svc)
y = self_training_model.fit(iris.data, iris.target)
print(y)

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
'''
classifier = KNeighborsClassifier
clf = classifier(n_neighbors=3)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)
print(X,y)
'''

# select features
X, y = load_iris(return_X_y=True, as_frame=True)
feature_names = X.columns
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=2)
sfs.fit(X, y)
print("Features selected by forward sequential selection: "
      f"{feature_names[sfs.get_support()].tolist()}")

from sklearn.datasets import fetch_covtype
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.linear_model import LogisticRegression
'''
X, y = fetch_covtype(return_X_y=True)

#ValueError: EOF: reading array data, expected 262144 bytes got 227911


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000,
                                                    test_size=10000,
                                                    random_state=42)

print(X_train, X_test, y_train, y_test )

# LogisticRegression
linear_baseline = make_pipeline(MinMaxScaler(), LogisticRegression(max_iter=1000))
linear_baseline.fit(X_train, y_train).score(X_test, y_test)

# PolynomialCountSketch + LogisticRegression
pipe = make_pipeline(MinMaxScaler(),
                     PolynomialCountSketch(degree=2, n_components=300),
                     LogisticRegression(max_iter=1000))
print(pipe.fit(X_train, y_train).score(X_test, y_test))
'''

from sklearn.datasets import fetch_california_housing
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']

from sklearn.tree import DecisionTreeRegressor

n_samples, n_features = 1000, 20
rng = np.random.RandomState(0)
X = rng.randn(n_samples, n_features)
# positive integer target correlated with X[:, 5] with many zeros:
y = rng.poisson(lam=np.exp(X[:, 5]) / 2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)
regressor = DecisionTreeRegressor(criterion='poisson', random_state=0)
print(regressor.fit(X_train, y_train))

# https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_coclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-coclustering-py
from matplotlib import pyplot as plt

from sklearn.datasets import make_biclusters
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

data, rows, columns = make_biclusters(
    shape=(300, 300), n_clusters=5, noise=5,
    shuffle=False, random_state=0)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralCoclustering(n_clusters=5, random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.3f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.show()



# https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html
#sphx-glr-auto-examples-bicluster-plot-spectral-biclustering-py
from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering

n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10,
    shuffle=False, random_state=0)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# shuffle clusters
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                             random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.1f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(np.outer(np.sort(model.row_labels_) + 1,
                     np.sort(model.column_labels_) + 1),
            cmap=plt.cm.Blues)
plt.title("Checkerboard structure of rearranged data")

plt.show()