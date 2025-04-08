from sklearn.datasets import load_iris

# Завантажуємо дані про іриси
iris_dataset = load_iris()

# Виводимо основні ключі набору даних
print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))

# Виводимо частину опису набору даних
print(iris_dataset['DESCR'][:193] + "\n...")

# Виводимо назви цільових класів
print("Назви відповідей: {}".format(iris_dataset['target_names']))

# Виводимо назви ознак
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))

# Виводимо тип масиву даних
print("Тип масиву data: {}".format(type(iris_dataset['data'])))

# Виводимо форму масиву даних (скільки рядків і стовпців)
print("Форма масиву data: {}".format(iris_dataset['data'].shape))

# Виводимо тип масиву цільових значень
print("Тип масиву target:{}".format(type(iris_dataset['target'])))

# Виводимо відповіді для кожного класу
print("Відповіді:\n{}".format(iris_dataset['target']))
