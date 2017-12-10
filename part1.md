> — Eh bien, mon prince. Gênes et Lucques ne sont plus que des apanages, des поместья, de la famille Buonaparte. Non, je vous préviens que si vous ne me dites pas que nous avons la guerre, si vous vous permettez encore de pallier toutes les infamies, toutes les atrocités de cet Antichrist (ma parole, j'y crois) — je ne vous connais plus, vous n'êtes plus mon ami, vous n'êtes plus мой верный раб, comme vous dites 1. Ну, здравствуйте, здравствуйте. Je vois que je vous fais peur 2, садитесь и рассказывайте.

#ТОМ ПЕРВЫЙ

###ЧАСТЬ ПЕРВАЯ. Анна Каренина

Недавно на хабре наткнулся на эту статью https://habrahabr.ru/post/342738/. И захотелось написать про word embeddings, python, gensim и word2vec. В этой части я постараюсь рассказать о обучении базовой модели w2v.

Итак, приступаем.

* Качаем anaconda. Устанавливаем.
* Еще нам пригодится C/C++ tools от visual studio.
* Теперь устанавливаем gensim. Именно для него нам и нужен c++.
* Устанавливаем nltk.
* При установке не забудьте качать библиотеки для Anaconda, а не для стандартного интерпретатора. Иначе все кончится крахом.
* Качаем [Анну Каренину](http://modernlib.ru/books/tolstoy_lev_nikolaevich/anna_karenina/) в TXT.
* Советую открыть файл и вырезать оттуда рекламу и заголовки. Потом сохранить в формате ```utf-8```.
* Можно приступать к работе.


<cut />Первым делом надо скачать данные для nltk.
```python
import nltk
nltk.dwonload()
```
В открывшемся окошке выбираем все, и идем пить кофе. Это займет около получаса.
По умолчанию в библиотеке русского языка нет. Но умельцы все сделали за нас. Качаем https://github.com/mhq/train_punkt и извлекаем все в папку
```C:\Users\<username>\AppData\Roaming\nltk_data\tokenizers\punkt ``` и
```C:\Users\<username>\AppData\Roaming\nltk_data\tokenizers\punkt\PY3```.


Nltk мы будем использовать для разбивки текста на предложения, а предложений на слова. К моему удивлению, все это работает довольно быстро. Ну хватит настроек, пару уже написать хоть строчку нормального кода.
Создаем папку где будут скрипты и данные. Создаем enviroment.
```bash
conda create -n tolstoy-like
```
Активируем.
```bash
activate tolstoy
```
Туда же кидаем текст. Назовем файл ```anna.txt```
Для обладателей PyCharm можно просто создать проект и в качестве интерпретатора выбрать анаконду, не создавая окружения.

Создаем скрипт ```train-I.py```.

* Подключаем зависимости.
```python
# -*- coding: utf-8 -*-
# imports
import gensim
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
```

* Считываем текст.
```python
# load text
text = open('./anna.txt', 'r', encoding='utf-8').read()
```

* Теперь очередь токенизатора русских предложений.
```python
def tokenize_ru(file_text):
    # firstly let's apply nltk tokenization
    tokens = word_tokenize(file_text)

    # let's delete punctuation symbols
    tokens = [i for i in tokens if (i not in string.punctuation)]

    # deleting stop_words
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...'])
    tokens = [i for i in tokens if (i not in stop_words)]

    # cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens
```
На этом остановимся по подробнее. В первой строчке мы разбиваем предложение (строку) на слова (массив строк). Затем удаляем пунктуацию, которую nltk, почему-то выносит как отдельное слово. Теперь стоп-слова. Это такие слова, от которых нашей модели пользе не будет, они лишь будут сбивать ее с основного текста. К ним относят междометия, союзы и некоторые местоимения, а также любимые некоторыми слова-паразиты. Затем убираем кавычки которых в этом романе через край.

* Теперь разбиваем текст на предложения, а предложения на массив слов.
```python
sentences = [tokenize_ru(sent) for sent in sent_tokenize(text, 'russian')]
```

* Для интереса выведем количество предложений и парочку из них.
```python
print(len(sentences))  # 20024
print(sentences[200:209])  # [['Она', 'чувствовала', 'боится', 'боится', 'предстоящего', 'свидания'],...]
```

* Теперь начинаем обучать модель. Не бойтесь это не займет и получасу - 20024 предложения для gensim просто расплюнуть.
```python
# train model
model = gensim.models.Word2Vec(sentences, size=150, window=5, min_count=5, workers=4)
```

* Теперь сохраняем модель в файл.
```python
# save model
model.save('./w2v.model')
print('saved')
```

Сохраняем файл. Чтобы запустить, тем кто работает в PyCharm или Spyder достаточно нажать run. Кто пишет вручную с блокнота или другого редактора придется запустить Anaconda Promt (для этого достаточно вбить это в поиск в меню), перейти в директорию со скриптом и запустить командой
```bash
python train-I.py
```
Готово. Теперь вы можете с гордостью сказать, что обучали word2vec.


###ЧАСТЬ ВТОРАЯ. Война и Мир

Как бы мы не старались, но Анны Каренины для обучении модели мало. Поэтому воспользуемся вторым произведением автора - Война и Мир.

Скачать можно [отсюда](http://vojnaimir.ru/download.html), также в формате TXT. Перед использованием придется соединить два файла в один. Кидаем в директорию из первой главы, называем ```war.txt```. Одной из прелестью использования gensim является то, что любую загруженную модель можно доучить с новыми данными. Этим мы и займемся.
Создаем скрипт ```train-II.py```

* Думаю, что эта часть не нуждается в объяснениях, так как в ней нет ничего нового.
```python
# -*- coding: utf-8 -*-
# imports
import gensim
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# load text
text = open('./war.txt', 'r', encoding='utf-8').read()
def tokenize_ru(file_text):
    # firstly let's apply nltk tokenization
    tokens = word_tokenize(file_text)

    # let's delete punctuation symbols
    tokens = [i for i in tokens if (i not in string.punctuation)]

    # deleting stop_words
    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...'])
    tokens = [i for i in tokens if (i not in stop_words)]

    # cleaning words
    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens
# tokenize sentences
sentences = [tokenize_ru(sent) for sent in sent_tokenize(text, 'russian')]
print(len(sentences))  # 30938
print(sentences[200:209])  # [['Он', 'нагнув', 'голову', 'расставив', 'большие', 'ноги', 'стал', 'доказывать', 'Анне', 'Павловне', 'почему', 'полагал', 'план', 'аббата', 'химера'],...]
```

* Затем загружаем нашу модель, и скармливаем ей новые данные.
```python
# train model part II
model = gensim.models.Word2Vec.load('./w2v.model')
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
```
Здесь я немного остановлюсь. ```total_examples``` устанавливает количество слов, в нашем случае это весь словарь модели (```model.corpus_count```) , включая новые. А ````epochs``` количество итераций. Честное слово, сам не знаю, что значит ```model.iter``` взял из документации. Кто знает, напишите, пожалуйста, в комментариях - исправлю.

* И снова сохраняем.
```python
# save model
model.save('./w2v-II.model')
print('saved')
```

Не забудьте запустить.

###ЭПИЛОГ. А где же тесты?

Их нет. И пока не будет. Модель еще не совсем совершенна, откровенно говоря, она ужасна. В следующей статье я обязательно расскажу как это исправить. Но вот вам на последок:

```python
# -*- coding: utf-8 -*-

# imports
import gensim

model = gensim.models.Word2Vec.load('./w2v-II.model')


print(model.most_similar(positive=['княжна', 'сестра'], negative=['князь'], topn=1))

```

**P.S**

> Вообще-то не все так плохо. Получившийся словарь содержит около 5 тысяч слов с их зависимостями и отношениями. В следующей статье я приведу более совершенную модель (15000 слов). Побольше расскажу о подготовке текста. И наконец в третьей части опубликую финальную модель и расскажу как с помощью нейронных сетей написать программу генерирующую текст в стиле Толстого.

Ссылки и используемая литература.

* https://habrahabr.ru/post/342738/
* https://github.com/mhq/train_punkt
* https://radimrehurek.com/gensim/models/word2vec.html
* https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
* https://rare-technologies.com/word2vec-tutorial/

### Успехов вам в машинном обучении.
Надеюсь моя статья вам хоть немного понравилась.