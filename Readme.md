##  Деплой модели детекции с помощью NVIDIA Triton Inference Server

![docker](https://img.shields.io/badge/docker-%232496ED.svg?&style=for-the-badge&logo=docker&logoColor=white)
![python](https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-vB900svg?style=for-the-badge&logo=NVIDIA&logoColor=white)

## Описание

В этом проекте реализовано два сервиса:
- **jupyter** - обучение, валидация и экспорт модели детекции + клиент для тритона.
- **triton** - инференс готовой модели детекции.

Блокноты:
- [Обучение, валидация и экспорт модели детекции](https://github.com/PitKoro/SberCloudTestTask/blob/main/detectron-train-export-to-torchscript/notebooks/train.ipynb)
- [Пример инференса обученной модели](https://github.com/PitKoro/SberCloudTestTask/blob/main/detectron-train-export-to-torchscript/notebooks/inference.ipynb)
- [Тест инференса модели в NVIDIA Triton Inference Server](https://github.com/PitKoro/SberCloudTestTask/blob/main/detectron-train-export-to-torchscript/notebooks/client_for_triton.ipynb)

Запуск **tensorboard** во время обучения находится в блокноте с обучением.

## Как запустить

1. Поставить [nvidia-docker и nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Скачать веса модели и перенести содержимое архива в `triton_server/models/faster_rcnn/1` - [Скачать веса]()
3. Запуск через docker-compose

```shell
docker compose up
```

## Тесты

В браузере пройти по локальному адресу `http://127.0.0.1:8888/lab/tree/notebooks/client_for_triton.ipynb` и выполнить подряд ячейки:
- В первой запуск скрипта на 100 запросов к серверу инференса.
- Остальные отвечают за единичный запрос и визуализацию результата.
