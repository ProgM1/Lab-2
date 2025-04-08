# Імпорт необхідних бібліотек
from fontTools.misc.textTools import tostr
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import numpy as np

# Завантажуємо дані з відкритого джерела
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Перевірка розміру завантаженого набору даних
print(dataset.shape)

# Виведення перших 20 рядків для перегляду структури даних
print(dataset.head(20))

# Опис статистичних характеристик всіх атрибутів
print(dataset.describe())

# Підрахунок кількості елементів в кожному класі
print(dataset.groupby('class').size())

# Візуалізація за допомогою діаграми розмаху для перевірки варіацій атрибутів
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Створення гістограми для розподілу значень атрибутів
dataset.hist()
pyplot.show()

# Побудова матриці розсіювання для вивчення залежностей між атрибутами
scatter_matrix(dataset)
pyplot.show()

# Розподіляємо дані на тренувальну та тестову вибірки
array = dataset.values
X = array[:,0:4]  # Вибір усіх ознак
y = array[:,4]    # Вибір цільової змінної (класу)

# Розподіл даних на тренувальну і тестову вибірки
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Список алгоритмів для тестування
models = []
models.append(('LR', OneVsRestClassifier(LogisticRegression())))  # Логістична регресія
models.append(('LDA', LinearDiscriminantAnalysis()))  # Лінійний дискримінантний аналіз
models.append(('KNN', KNeighborsClassifier()))  # Алгоритм найближчих сусідів
models.append(('CART', DecisionTreeClassifier()))  # Дерево рішень
models.append(('NB', GaussianNB()))  # Байєсівський класифікатор
models.append(('SVM', SVC(gamma='auto')))  # Метод опорних векторів (SVM)

# Оцінка кожної моделі за допомогою крос-валідації
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

# Порівняння результатів різних алгоритмів за допомогою діаграми
pyplot.boxplot(results, tick_labels=names)
pyplot.title('Порівняння алгоритмів')
pyplot.show()

# Тренування та тестування моделі SVM
model = SVC(gamma='auto')
model.fit(X_train, Y_train)

# Оцінка результатів на тестових даних
predictions = model.predict(X_validation)

# Виведення точності прогнозу та інших показників
print("Оцінка прогноза:")
print(f"Точність: {accuracy_score(Y_validation, predictions):.6f}")
print("Матриця плутанини:")
print(confusion_matrix(Y_validation, predictions))
print("Звіт про класифікацію:")
print(classification_report(Y_validation, predictions))

# Прогнозування для нового входження за допомогою натренованої моделі
new_data = np.array([[5.0, 3.0, 1.5, 0.2]])

# Використовуємо модель SVM для прогнозування нового прикладу
svm_model = [model for name, model in models if name == 'SVM'][0]
svm_model.fit(X_train, Y_train)

# Прогноз для нового запису
prediction = svm_model.predict(new_data)
print("////////////////////////////")
print("Прогноз: " + str(prediction[0]))
print("////////////////////////////")


