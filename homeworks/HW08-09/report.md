# HW08-09 Report

## 1. Выбранный датасет
В данной версии ноутбука использовался synthetic dataset формата [1, 28, 28] с 10 классами.
Это было сделано из-за недоступности внешних серверов загрузки датасетов.

## 2. Архитектура модели
Использовалась MLP-модель с двумя скрытыми слоями.
В экспериментах сравнивались базовая версия, Dropout, BatchNorm и EarlyStopping.

## 3. Эксперименты части A
Проведены эксперименты E1-E4.
Лучшая модель выбиралась по `val_accuracy`.

См.:
- `artifacts/runs.csv`
- `artifacts/figures/curves_best.png`

## 4. Эксперименты части B
Проведены эксперименты O1-O3:
- слишком большой learning rate
- слишком маленький learning rate
- SGD + momentum + weight decay

См.:
- `artifacts/runs.csv`
- `artifacts/figures/curves_lr_extremes.png`

## 5. Финальная модель
Лучшая модель сохранена в:
- `artifacts/best_model.pt`
- `artifacts/best_config.json`

## 6. Вывод
Dropout/BatchNorm и EarlyStopping влияют на поведение модели при обучении.
Слишком большой learning rate приводит к нестабильности,
слишком маленький — к очень медленному обучению.