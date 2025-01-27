# Conda



<details markdown="1">
  <summary>Table of Contents</summary>

-   [TL;DR](#tldr)
-   [Conda installation](#conda-installation)
-   [Cheat sheet](#conda-cheat-sheet)
-   [Version Specification](#conda-version-specs)
-   [Нюансы совместной работы с pip](#pip-and-conda)
-   [How conda works](#conda)
    * [Downloading and processing index metadata](#conda-metadata)
    * [Reducing the index](#conda-reducing-index)
    * [Подробнее о SAT Problem](#conda-sat)
-   [Чтиво](#chtivo)
</details>




<a id="tldr"></a>
<a id="consistency"></a>
## TL;DR
- Используешь conda-forge?
    - Тогда используй conda-metachannel чтобы понизить размерность алгоритма разрешения зависимостей.
- Указываешь широкие спецификации пакета?
    - Ставь спецификации на версии конкретными. Вместо “numpy”, используй “numpy=1.15” или даже лучше так: “numpy=1.15.4”
- Начинает бесить бесконечный “verifying transaction”?
    - `conda config –set safety_checks disabled`
- Getting strange mixtures of defaults and conda-forge?
    - `conda config –set channel_priority strict`
    - Это тоже ускорит процесс, испключая возможные смешанные решения
- Заметили, что conda начинает работать все медленнее и медленнее?
    - Создайте [fresh environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). С ростом размера виртуального окружения, они становятся все сложнее и слоджнее для разрешения зависимостей

<a id="conda-installation"></a>
<a id="consistency"></a>
## Conda installation
Идем в конец странички https://www.anaconda.com/download, и выбираем нужный дистрибутив, например https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh

Далее
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
bash Anaconda3-2023.07-2-Linux-x86_64.sh
```

Не забываем поставить галочку для автоматического conda init, чтобы не лезть в path.

<a id="conda-cheat-sheet"></a>
<a id="consistency"></a>
## Cheat sheet

```bash
# Обновить конду (не пакеты в окружении)
conda update conda

# Принт всех установленных окружений
conda env list

# Принт всех пакетов в текущем окружении
conda list

# Сохранить окружение в файл
conda list --explicit > bio-env.txt

# Воссоздать окружение из файла
conda env create --file bio-env.txt

# Новое окружение
conda create --name MY_AWESOME_ENV

# Активировать окружение
conda activate MY_AWESOME_ENV

# Деактивировать окружение
conda deactivate

# Удалить окружение
conda remove --name MY_AWESOME_ENV --all

# Установить пакетик в окружение
conda install scikit-learn

# Установить версию питона в окружение
conda install python=3.10

# Обновить пакетик в текущем окружении
conda update scikit-learn

# Установить напрямую с pypi
pip install scikit-learn
```



<a id="conda-version-specs"></a>
<a id="consistency"></a>
## Version specification
| Constraint type           | Specification          | Result                            |
|---------------------------|------------------------|-----------------------------------|
| Fuzzy                     | `numpy=1.11`           | 1.11.0, 1.11.1, 1.11.2, 1.11.18 etc. |
| Exact                     | `numpy==1.11`          | 1.11.0                            |
| Greater than or equal to  | `"numpy>=1.11"`        | 1.11.0 or higher                  |
| OR                        | `"numpy=1.11.1\|1.11.3"`| 1.11.1, 1.11.3                    |
| AND                       | `"numpy>=1.8,<2"`      | 1.8, 1.9, not 2.0                 |




<a id="pip-and-conda"></a>
<a id="consistency"></a>
## Нюансы совместной работы pip и conda
- Конфликты зависимостей
    - Conda и pip разрешают зависимости по-разному. При установке пакетов через оба инструмента в одном окружении могут возникнуть несовместимые версии пакетов.

- Дублирование пакетов
    - Если пакет установлен и через Conda, и через pip, это может привести к дублированию пакетов, что увеличит размер окружения и может вызвать путаницу.

- Бинарная совместимость
    - Conda пакеты предоставляют бинарно совместимые библиотеки для определенных платформ. Установка пакетов через pip может привести к установке бинарных файлов, которые не тестировались или не оптимизированы для вашей платформы или окружения Conda.

- Как pip разрешает зависимости
    - pip проверяет зависимости указанные в setup.py или pyproject.toml пакета
    - Если указана конкретная версия зависимости, pip попытается установить именно ее. Если указан диапазон версий, pip выберет наиболее подходящую версию в этом диапазоне.
    - В отличие от Conda, который пытается решить NP-полную задачу k-SAT для поиска оптимального набора пакетов, pip просто следует цепочке зависимостей и пытается установить последние доступные версии, что может иногда приводить к несовместимостям.


<a id="conda"></a>
<a id="consistency"></a>
# How conda works?

1. Downloading and processing index metadata
1. Reducing the index
1. Expressing the package data and constraints as a SAT problem
1. Running the solver
1. Downloading and extracting packages
1. Verifying package contents
1. Linking packages from package cache into environments



<a id="conda-metadata"></a>
<a id="consistency"></a>
## Downloading and processing index metadata

Метаданные в conda поступают из JSON-файлов (либо repodata.json, либо его сжатой формы, repodata.json.bz2). В отличие от многих пакетных менеджеров, репозитории Anaconda обычно не фильтруют или не удаляют старые пакеты из индекса. Это хорошо, так как старые окружения легко воссоздаются. Но это также означает, что метаданные индекса растут, и conda становится медленнее по мере увеличения пакетов. Либо пакеты со временем переходят в архивные каналы, либо сервер должен представлять урезанный вид всех доступных пакетов.


<details markdown="1">
<summary>Conda-metachannel</summary>

[Conda-metachannel](https://medium.com/@marius.v.niekerk/conda-metachannel-f962241c9437) — это комьюнити-проект, начатый Marius van Niekerk, который пытается уменьшить размер repodata, передаваемого в conda. Это по сути server-side решение, которое кешируется и используется для многих пользователей, уменьшая workload на компьютере пользователя. Всё это происходит за кулисами, и conda-metachannel предоставляет файл repodata.json, который не отличается, поэтому от conda не требуется спецподдержка. Эти идеи будут ключевыми для будущего развития репозиториев, предлагаемых Anaconda, так как они позволяют экосистеме расти без замедления conda. Конкретных разработок на данный момент нет, кроме conda-metachannel, но эти идеи будут частью будущего развития.
</details>


После загрузки метаданных, conda загружает JSON-файл в память и создаёт объекты для каждого пакета.
Для особенно больших каналов, таких как conda-forge, этот этап может занимать много времени. Например, рассмотрим создание простого окружения без кешированного индекса:

```bash
conda clean -iy && CONDA_INSTRUMENTATION_ENABLED=1 conda create -n test_env –dry-run python numpy
```
Добавление каналов conda-forge существенно увеличивает время, затрачиваемое на создание индекса, в то время как использование conda metachannel экономит много времени:

| | Defaults only | Conda-forge | Bioconda + Conda-forge | Conda-forge with metachannel |
|-------------------|:----------:|:---------:|:------------------------:|:--------------------------:|
| **Metadata collection time** | 2.80s | 8.33s | 10.23s | 3.41s |






<a id="conda-reducing-index"></a>
<a id="consistency"></a>
## Reducing the index

На текущем этапе repodata, вероятно, содержит много данных о пакетах, которые не используются на этапе решения зависимостей. Следующий этап, выражение данных о пакетах и ограничений как проблемы SAT, довольно ресурсоемкий, и фильтрация ненужных пакетов может cыкономить время. Conda начинает с явных спецификаций (если есть), предоставленных пользователем. Затем Conda рекурсивно проходит через зависимости этих спецификаций, чтобы создать полный набор пакетов которые могут быть использованы в любом окончательном решении. Пакеты, которые не участвуют явно или как зависимость, обрезаются на этом этапе. На этом шаге рекомендуется быть максимально конкретным в своих спецификациях пакетов (то есть указывать максимально конеретные версии).

Кстати одна из оптимизаций, проведенных в conda 4.6, заключалась в том, чтобы сделать это обрезание более агрессивным. При рекурсии Conda через эти зависимости specs не могут "расширяться" от ограничений, выраженных в явных зависимостях. Например, метапакет anaconda состоит из всех точных ограничений (указаны и версия, и строка сборки). Зависимость zlib этого метапакета выглядит примерно как "zlib 1.2.11 habc1234_0".


Это деликатный баланс: если Conda фильтрует слишком агрессивно, пользователи могут не видеть ожидаемых решений, и причины, по которым они не видят эти решения, могут быть неясны. По этой причине conda делает два осторожных учета:

- не пытается фильтровать фактические значения диапазонов версий, ни строки сборки.
- сортирует коллекцию зависимостей по убыванию версии, так что новейшие пакеты оказывают наибольшее влияние на то, насколько расширение допустимо или не допустимо.





<a id="conda-sat"></a>
<a id="consistency"></a>
## Подробнее о SAT Problem (Задача выполнимости булевых формул)
[wiki](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)

SAT-проблема или задача о выполнимости булевых формул является важной и сложной задачей. Она состоит в том, чтобы определить, можно ли найти такое значение переменных, при которых булева формула будет истинной. Если такое значение существует, формула считается выполнимой, в противном случае – не выполнимой.

SAT-проблема является NP-полной.


$$ \text{{SAT}}(F) = \begin{cases} 
\text{{TRUE}}, & \text{если } \exists x_1, x_2, \dots, x_n \text{ такое что } F(x_1, x_2, \dots, x_n) = \text{{TRUE}} \\
\text{{FALSE}}, & \text{иначе}
\end{cases} $$


##### Пример
$$ \text{{SAT}}(a \land \neg b) = \begin{cases} 
\text{{TRUE}}, & \text{если } a = \text{{TRUE}} \text{ и } b = \text{{FALSE}} \\
\text{{FALSE}}, & \text{иначе}
\end{cases} $$

##### Проблема k-SAT

$$ \text{{3-SAT}}(F) = \begin{cases} 
\text{{TRUE}}, & \text{если } \exists x_1, x_2, \dots, x_n \text{ такое что } F(x_1, x_2, \dots, x_n) = \text{{TRUE}} \\
\text{{FALSE}}, & \text{иначе}
\end{cases} $$

k-SAT: Каждый дизъюнкт в конъюнктивной нормальной форме содержит ровно k литералов. 
Пример для k=3: $(x_1 \lor x_2 \lor x_4) \land (y_1 \lor x_2  \lor x_5)$

##### Применительно к conda
Сначала Conda загружает и обрабатывает метаданные для доступных пакетов. Затем она сокращает индекс, оставляя только те пакеты, которые соответствуют требованиям пользователя и зависимостям пакета. Далее, Conda формулирует проблему установки пакета как SAT-проблему. Этот шаг включает представление зависимостей и конфликтов между пакетами в виде булевых формул. Затем Conda использует SAT-решатель для определения того, существует ли набор пакетов, который удовлетворяет всем требованиям и ограничениям. Если такой набор существует, Conda продолжает процесс установки пакета, загружая и извлекая необходимые пакеты, проверяя их содержимое и связывая их с соответствующими окружениями.

Кроме того conda использует множество эвристик.
Conda не просто ищет удовлетворимость для SAT – напротив, нам нужно не абстрактное, а вполне конкретное решение, например, самый новый пакет, или пакет из канала с максимальным приоритетом. Поэтому Conda присваивает скоры пакетам с одинаковым именем. Эти оценки учитывают приоритет канала, различия в версиях, номер сборки и таймстемп.




<a id="chtivo"></a>
<a id="consistency"></a>
## Дополнительное чтиво
- https://www.anaconda.com/blog/understanding-and-improving-condas-performance
- https://docs.conda.io/projects/conda/en/4.14.x/dev-guide/deep-dive-solvers.html
