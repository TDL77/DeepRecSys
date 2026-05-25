
# Теоретический минимум

По каждому термину нужно уметь дать короткое определение, объяснить зачем он нужен в рекомендательных системах и привести простой пример.

### Общие понятия
- многостадийная рекомендательная система
- генерация кандидатов
- ранжирование
- impression
- холодный старт
- тяжелый хвост
- feedback loop
- distribution shift

### Оценка качества и смещения
- методы деления на train/test split в рекомендациях
- Recall@K
- NDCG@K
- position bias

### Candidate generation / retrieval
- двухбашенная модель
- поиск ближайших соседей / ANN search
- sampled softmax loss
- in-batch negatives
- temperature
- logQ correction

### Ranking
- pointwise ranking loss
- pairwise ranking loss
- listwise ranking loss
- hashing trick
- факторизационные машины
- кросс-слой / DCN-v2
- multi-task learning
- knowledge distillation

### Sequential recommenders
- next-item prediction
- target-aware attention

### Graph-based recommendations
- link prediction
- случайные блуждания
- GraphSAGE
- LightGCN

### Bandits и RL
- exploration-exploitation dilemma
- regret
- epsilon-greedy
- UCB
- Thompson sampling
- REINFORCE

### Generative recommendations
- генеративная модель
- авторегрессивная модель
- generative retrieval
- item-action interleaving
- semantic ID

# Экзаменационные билеты

В билетах модели и статьи используются как примеры. Не требуется помнить все архитектурные детали каждой модели. Важно понимать, какую задачу решает подход, какая у него основная идея, как он обучается, как оценивается и какие у него ограничения.

## 1. Основы рекомендательных систем

**Билет 1. Что такое рекомендательная система: цели, участники, базовые алгоритмы**  
Нужно уметь объяснить, зачем нужны рекомендательные системы, какие цели могут быть у пользователя, платформы и авторов контента, а также привести примеры простых алгоритмов: коллаборативная фильтрация, матричная факторизация.

**Билет 2. Многостадийная архитектура рекомендательной системы**  
Почему нельзя сразу ранжировать все айтемы тяжелой моделью. Какие бывают стадии и чем они отличаются.

**Билет 3. ML-дизайн рекомендательной системы**  
Как формулировать задачу: выбор целевой метрики, выбор данных и задачи обучения на разных стадиях.

**Билет 4. Оффлайн-оценка качества рекомендаций**  
Правильное деление на train и валидацию, основные оффлайн-метрики для различных стадий рексистемы. 

## 2. Candidate generation / retrieval

**Билет 5. Двухбашенные модели для генерации кандидатов**  
Как выглядит модель. Как используем её в продакшне, почему такая арх-ра хорошо подходят для поиска кандидатов среди большого каталога. 

**Билет 6. Обучение retrieval-моделей: softmax, sampled softmax, InfoNCE, in-batch negatives**  
Основная функция потерь для обучения retrieval моделей (sampled softmax). Почему не подходит ранжирующий лосс (про фолдинг). Почему не можем использовать глобальный softmax по айтемам, как его аппроксимируем, какие есть потенциальные источники негативов, что такое температура и для чего она нужна. 

**Билет 7. Sampling bias и logQ correction**
Почему при sampled/in-batch negatives возникает sampling bias и как его исправляет logQ correction.

**Билет 8. ID-based embeddings, content encoders и inductive bias**  
Обучаемые эмбеддинги, их недостатки и преимущества (e.g., память, холодный старт, тяжелый хвост, переобучение). Контентные энкодеры как полезный inductive bias

**Билет 9. Подходы за пределами классического two-tower retrieval**  
Multi-interest retrieval, mixture-of-logits, GPU retrieval, generative retrieval как альтернатива двухбашенным моделям.

## 3. Ranking

**Билет 10. Обучение ранжирования на impression-логах**  
Что такое impression, какие целевые метки используются в зависимости от бизнес-модели рексистемы. Что считается негативами. Pointwise/pairwise/listwise постановки. 

**Билет 11. Преимущества нейросетевых моделей для задачи ранжирования**
Какие у нейросетевого ранжирования есть преимущества, почему недостаточно градиентного бустинга. Когда стоит использовать нейросетевое ранжирование. В чем принципиальная разница с нейросетевым retrieval 

**Билет 12. Категориальные и вещественные признаки в neural ranking**  
Обучаемые эмбеддинги, hashing trick и unified эмбеддинги, методы обработки вещественных признаков (включая PLE)

**Билет 13. Моделирование взаимодействий признаков для нейросетевого ранжирования**
Кросс-признаки, линейные модели, факторизационные машины, явное и неявное моделирование взаимодействия признаков (почему недостаточно простого MLP), кросс-слои из DCN-v2. 

**Билет 14. Attention и transformer-like слои для feature interactions**  
Как attention можно применять к набору признаков. AutoInt, HiFormer, OneTrans, MixFormer как примеры подходов к моделированию взаимодействий признаков.

**Билет 15. Multi-task learning, debiasing и distillation в ranking** 
Многозадачное обучение (когда у нас сразу несколько сигналов) и архитектуры для него (e.g., MMoE). ESMM (одновременное моделирование CTR/CVR), позиционный bias в ранжировании и способы борьбы с ним. Knowledge distillation для улучшения ранжирующих моделей

## 4. Sequential recommenders

**Билет 16. Моделирование пользователя в ранжирующих моделях**  
Как использовать историю пользователя в ранжирующих моделях. Пулинг событий, target-aware attention, DIN, BST, TransAct. Отличие от next-item prediction.

**Билет 17. Sequential recommenders**
Next-item prediction. Академический подход, GRU4Rec, SASRec, BERT4Rec-like постановки. Индустриальный подход - PinnerFormer.

## 5. Graph-based recommendations

**Билет 18. Графовые модели на user-item графе: NGCF, LightGCN**  
Двудольный user-item граф, задача link prediction, агрегация соседей, NGCF & LightGCN.

**Билет 19. GraphSAGE и PinSage для рекомендаций**  
Индуктивные графовые модели, neighbor sampling, GraphSAGE, PinSage, масштабирование GNN на больших графах и применение в рекомендациях.

**Билет 20. Graph representation learning и случайные блуждания в рекомендациях**  
Случайные блуждания, обучение представлений на больших и гетерогенных графах, TWHIN/TTGL-like системы как примеры.

## 6. Reinforcement learning

**Билет 21. Feedback loop, exposure bias и смещения в данных**  
В чем заключается feedback loop и почему он проблематичен. Popularity bias, exposure bias. Что такое filter bubble и echo chamber.

**Билет 22. Многорукие бандиты и exploration/exploitation**  
Формализация bandit-задачи, regret, epsilon-greedy, UCB, Thompson sampling, зачем нужен exploration в рекомендательных системах.

**Билет 23. RL-формализация рекомендаций и REINFORCE**  
State, action, reward, policy, trajectory, return, policy gradient, REINFORCE, all-actions formulation, off-policy correction. Как эти сущности маппятся на задачу рекомендаций. Сложности применения RL в рекомендациях.

## 7. Generative recommendations и LLM

**Билет 24. Определение генеративной модели**  
Что такое генеративные модели, авторегрессивное моделирование. Примеры генеративных моделей в рекомендациях.

**Билет 25. Генеративное ранжирование на примере HSTU-like моделей**  
Чем HSTU-like подход отличается от классического ranking-подхода с DLRM-like моделью.

**Билет 26. Generative retrieval, semantic IDs, TIGER и OneRec**  
Построение и применение semantic IDs, задача generative retrieval. TIGER-like (и OneRec-like) архитектура. Проблемы и вызовы семантических идентификаторов.

**Билет 27. Conversational recommender systems: PLUM, NEO**
Недостатки модели P5 (почему у нее было плохое качество). Разговорные рекомендации на примере моделей PLUM и NEO - основные архитектурные элементы, различные стадии обучения моделей (LLM-RecSys alignment, generative retrieval fine-tuning, etc)