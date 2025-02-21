# Kolmogorov Image Compressor

English Version
---------------

### Project Overview

This project implements an image compression algorithm inspired by the Kolmogorov–Arnold representation theorem. It combines modern techniques for efficient image compression and reconstruction, including:

*   **Fourier Features** – mapping input coordinates into a sinusoidal feature space to enhance network learning.
*   **Perceptual Loss** – using a pre-trained VGG16 network to preserve high-quality visual features in the reconstructed image.
*   **Adversarial Training** – using a discriminator to improve the realism of the reconstructed image.
*   **Fine-Tuning** – additional training on specific images to enhance reconstruction quality.
*   **Post-Processing** – functions for super-resolution, denoising, contrast enhancement, and JPEG compression evaluation (using PSNR and SSIM).

### Key Features

*   **Generator Architecture:** Utilizes multiple Kolmogorov blocks, each comprising a Psi-network, Phi-network, and Residual blocks.
*   **Integration of Fourier Features:** Transforms input data into a more expressive feature space.
*   **Adversarial Training:** Incorporates a discriminator to boost the visual plausibility of the output.
*   **Configurable Parameters:** Easily adjust model parameters such as the number of blocks, hidden layer size, training parameters, and loss function weights.
*   **End-to-End Workflow:** From image loading and preprocessing to reconstruction, post-processing, and quality evaluation.

### Requirements

To run the project, ensure you have:

*   Python 3.x
*   PyTorch and torchvision
*   NumPy
*   Pillow (PIL)
*   OpenCV (cv2)
*   Matplotlib
*   Scikit-Image

### Installation

1.  **Clone the repository and navigate to the project directory:**
    
    ```bash
    git clone <YOUR_REPO_URL>
    cd <repository_name>
    ``` 
    
2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ``` 
    
### Project Structure

*   **compress.py** – the main script, which includes:
    *   **Classes:**
        *   `KolmogorovImageCompressor` – the generator for image compression.
        *   `PsiNetwork`, `PhiNetwork`, `ResidualBlock` – network blocks for building the generator.
        *   `ImageDiscriminator` – the discriminator for adversarial training.
    *   **Functions:**
        *   Loading and preprocessing images.
        *   Creating a coordinate grid.
        *   Training the model with a composite loss function (MSE, L1, perceptual, and adversarial loss) via `compress_image()`.
        *   Fine-tuning the model via `fine_tune_image()`.
        *   Decompressing and reconstructing the image via `decompress_image()`.
        *   Post-processing (super-resolution, denoising, contrast enhancement) via `postprocess_image()`.
        *   JPEG compression and quality evaluation (PSNR, SSIM) via `compress_to_jpeg()` and `evaluate_compression()`.

### Usage

#### Running the Script

To start the project, execute the following command in your terminal:

`python compress.py` 

#### Interactive Workflow

*   **Providing the Image Path:**  
    When the script starts, it will prompt you to enter the path of the image to be used for training:
    
    `Enter the image path: example.jpg` 
    
*   **Training:**  
    If the model file (`compressed_model.pth`) is not found, training will begin from scratch. Loss values (MSE, L1, perceptual, adversarial) are printed every 1000 epochs.
*   **Fine-Tuning:**  
    After the initial training, the script will prompt:
    
    `Perform fine-tuning? (y/n):` 
    
    Type `y` to start the fine-tuning process.
*   **Reconstruction and Post-Processing:**  
    After training or fine-tuning:
    *   The model reconstructs the image and displays it.
    *   Post-processing (super-resolution, denoising, contrast enhancement) is applied and the result is shown.
    *   JPEG compression is applied to the original image, and quality metrics (PSNR, SSIM) are calculated and printed.

#### Full Workflow Example

### 1. Run the script
python compress.py

### 2. When prompted, enter the image path:
Enter the image path: example.jpg

### 3. During training, the console prints loss values, e.g.:
Epoch 0: Loss_G = 0.123456 (MSE=0.123456, L1=0.123456, Perc=0.123456, Adv=0.123456), Loss_D = 0.123456
...

### 4. After training is complete:
Model saved to compressed_model.pth

### 5. The script then prompts for fine-tuning:
Perform fine-tuning? (y/n):
### 6. Type "y" to start fine-tuning.

* Once training and fine-tuning are finished, the model reconstructs the image and displays it.
* Post-processing is applied, and the final image is shown.
*  Finally, JPEG compression is performed, and PSNR/SSIM metrics are printed.

## Testing and Configuration

*   **Testing:**  
    To test the compression and decompression routines, use sample images in the working directory and provide their correct paths when prompted.
*   **Configuration:**  
    All configurable parameters (e.g., `image_size`, `num_blocks`, `hidden_dim`, `fourier_mapping_size`, `fourier_scale`, number of epochs, learning rate, loss weights) are defined within the functions `compress_image()`, `decompress_image()`, and `fine_tune_image()`. These parameters can be adjusted for experimental purposes.

* * *

---

## Русская версия

### Описание проекта
Данный проект реализует алгоритм сжатия изображений, основанный на представлении, вдохновлённом теоремой Колмогорова–Арнольда. Проект объединяет современные методы для эффективного сжатия и реконструкции изображений, включая:

- **Fourier Features** – преобразование входных координат в синусоидальное пространство признаков для улучшения обучаемости сети.
- **Перцептивную потерю** – использование предобученной модели VGG16 для сохранения высококачественных визуальных характеристик реконструированного изображения.
- **Adversarial Training** – применение дополнительного дискриминатора для повышения реалистичности выходного изображения.
- **Fine-Tuning (дообучение)** – возможность дополнительного обучения модели на конкретных изображениях для улучшения результата.
- **Постобработка** – функции суперразрешения, шумоподавления и улучшения контраста, а также сравнение с JPEG-сжатием (расчёт PSNR и SSIM).

### Особенности проекта
- **Архитектура генератора:** Использование нескольких Kolmogorov-блоков, каждый из которых включает Psi-сеть, Phi-сеть и Residual-блоки.
- **Интеграция Fourier Features:** Преобразование входных данных для повышения выразительности признаков.
- **Adversarial Training:** Интеграция дискриминатора для повышения визуальной правдоподобности реконструкции.
- **Гибкая настройка:** Возможность изменения параметров модели, таких как число блоков, размер скрытого слоя, параметры обучения и коэффициенты для функций потерь.
- **Полный цикл обработки:** От загрузки и предобработки изображения до реконструкции, постобработки и оценки качества сжатия.

### Требования
Для работы проекта необходимы:
- Python 3.x
- PyTorch и torchvision
- NumPy
- Pillow (PIL)
- OpenCV (cv2)
- Matplotlib
- Scikit-Image

### Установка

1. **Клонирование репозитория и переход в директорию проекта:**

   ```bash
   git clone <URL_вашего_репозитория>
   cd <название_репозитория>
   ``` 

2.  **Установка зависимостей:**
    
    ```bash
    pip install -r requirements.txt
    ``` 
  

### Структура проекта

*   **compress.py** – основной скрипт, который включает:
    *   **Классы:**
        *   `KolmogorovImageCompressor` – генератор для сжатия изображений.
        *   `PsiNetwork`, `PhiNetwork`, `ResidualBlock` – сетевые блоки для построения генератора.
        *   `ImageDiscriminator` – дискриминатор для adversarial training.
    *   **Функции:**
        *   Загрузки и предобработки изображений (например, `load_image`).
        *   Создания координатной сетки (`create_coordinate_grid`).
        *   Обучения модели с комбинированной функцией потерь (MSE, L1, перцептивная и adversarial потери) – функция `compress_image()`.
        *   Дообучения (fine-tuning) модели – функция `fine_tune_image()`.
        *   Декомпрессии и реконструкции изображения – функция `decompress_image()`.
        *   Постобработки изображения (суперразрешение, шумоподавление, гистограммное выравнивание) – функция `postprocess_image()`.
        *   Сжатия изображения в формат JPEG и оценки качества сжатия (PSNR, SSIM) – функции `compress_to_jpeg()` и `evaluate_compression()`.

### Использование

#### Запуск скрипта

Для запуска проекта выполните в терминале:

`python compress.py` 

#### Интерактивная работа

*   **Ввод пути к изображению:**  
    При запуске скрипт запросит путь к изображению, например:
    
    `Введите путь к изображению: example.jpg` 
    
*   **Обучение модели:**  
    Если файл модели (`compressed_model.pth`) не найден, обучение будет производиться с нуля. Каждые 1000 эпох на консоли выводятся значения ошибок (например, MSE, L1, перцептивная и adversarial потери).
*   **Дообучение модели (Fine-Tuning):**  
    После завершения основного обучения скрипт предложит выполнить дообучение:
    
    `Выполнить дообучение модели? (y/n):` 
    
    Введите `y`, чтобы запустить процесс дообучения.
*   **Реконструкция и постобработка:**  
    По завершении обучения или дообучения:
    *   Модель восстанавливает изображение и выводит его на экран.
    *   Выполняется постобработка (суперразрешение, шумоподавление, улучшение контраста) и результат отображается в отдельном окне.
    *   Выполняется JPEG-сжатие исходного изображения, и рассчитываются PSNR и SSIM для оценки качества.

#### Пример полного цикла работы

1. Запуск скрипта

```bash
python compress.py
```

2. При запросе введите путь к изображению:
Введите путь к изображению: example.jpg

2. В процессе обучения на консоли выводятся данные, например:
```bash 
Epoch 0: Loss_G = 0.123456 (MSE=0.123456, L1=0.123456, Perc=0.123456, Adv=0.123456), Loss_D = 0.123456 
...
```

3. По завершении обучения:
Модель сохранена в compressed_model.pth

4. Затем запрос на дообучение:
Выполнить дообучение модели? (y/n):
5. Введите "y", чтобы запустить дообучение.

6. После обучения и дообучения модель реконструирует изображение, выполняется его постобработка, и отображается результат.
7. Производится JPEG-сжатие, а также рассчитываются PSNR и SSIM.` 

### Тестирование и настройка

*   **Тестирование:**  
    Используйте тестовые изображения, размещённые в рабочей директории, и указывайте корректные пути к ним.
*   **Настройка параметров:**  
    Параметры обучения (например, `image_size`, `num_blocks`, `hidden_dim`, `fourier_mapping_size`, `fourier_scale`, количество эпох, learning rate, веса функций потерь) определены в функциях `compress_image()`, `decompress_image()` и `fine_tune_image()`. Эти параметры можно изменять для проведения экспериментов.

* * *