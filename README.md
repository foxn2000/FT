## A5000 x 8基でLoraファインチューニングをするためのプログラム

mistralをLoraファインチューニングをするためだけのプログラムだけど、コードをかえれば何でもできるので、頑張ってください。

## 手順
・ライブラリ.txtの中にあるライブラリをインストール

・データセットをセッティング

・DeepSpeedのコマンドで実行

```
deepspeed --num_gpus=自分のGPUの数 test.py --deepspeed --deepspeed_config zero_train.json
```

・Marge.pyを実行し、Loraアダプターを適用したモデルを保存。
