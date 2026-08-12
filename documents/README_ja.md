# FasterSVC : 蒸留モデルとk近傍法に基づく高速な声質変換
(このリポジトリは実験段階のものです。内容は予告なく変更される場合があります。)

## モデル構造
![Architecture](../images/fastersvc_architecture.png)
デコーダーの構造はFastSVCやStreamVC, Hifi-GAN等を参考に設計。
未来の情報を参照しない"Causal"な畳み込み層を使用することで低遅延を実現。

## 特徴
- リアルタイム変換
- 低遅延 (約0.2秒程度、環境や最適化によって変化する可能性あり。)
- 位相とピッチが安定している (ソースフィルタモデルに基づく。)
- k近傍法による話者スタイル変換

## 必要なもの
- Python 3.10 以降
- PyTorch 2.0以降と GPU 環境
- フルスクラッチで訓練する場合は多数の人間の音声データを用意すること。(LJ SpeechやJVS コーパスなど。)

## インストール
1. このリポジトリをクローン
```sh
git clone https://github.com/AkaakuHub/fastersvc.git
```
2. 依存関係をインストール
```sh
pip3 install -r requirements.txt
```
## 事前学習モデルをダウンロードする
JVSコーパスで事前学習したモデルを[こちら](https://huggingface.co/uthree/fastersvc-jvs-corpus-pretrained)にて公開しています。content encoderとpitch estimatorは初期値として利用できます。修正後の単一励振源decoderとdiscriminatorはcheckpoint契約が異なるため、再学習が必要です。

## 事前学習
基礎的な音声変換を行うモデルを学習する。この段階では特定の話者に特化したモデルになるわけではないが、基本的な音声合成ができるモデルをあらかじめ用意しておくことで、少しの調整だけで特定の話者に特化したモデルを学習することができる。

以下に手順を示す。
1. 音声とF0を学習cacheへ前処理
```sh
python3 preprocess.py <dataset-directory> --output dataset_cache
```

2. ピッチ推定器を学習
WORLDのHarvestで求めた無加工の音高を、高速かつ並列に処理可能な1次元CNNへ蒸留する。既定では10,000step学習し、500stepごとに保存する。同じ`--training-state-path`を指定するとoptimizerとscalerを含む状態から再開する。
```sh
python3 train_pe.py --dataset-cache dataset_cache
```

3. コンテンツエンコーダーを学習。
初期層より入力話者の情報が少なく、音素内容を保持するHuBERT-baseの第9隠れ層を蒸留する。
```sh
python3 train_ce.py --dataset-cache dataset_cache
```

4. デコーダーを学習
デコーダーは、コンテンツ、250HzのA特性ラウドネス、有声／無声励振源から元の波形を再構築する。FastSVC論文のMulti-resolution STFT lossとLeast-squares adversarial lossを使用する。識別器は100,000stepから参加し、学習率は100,000stepごとに半減し、既定では600,000step学習する。

```sh
python3 train_dec.py --dataset-cache dataset_cache --fp16
```

## ファインチューニング
事前学習したモデルを、特定話者への変換に特化したモデルに調整することによって、より精度の高いモデルを製作することが可能です。この工程は事前学習と比べて非常に少ない時間で完了します。
1. 特定話者の音声ファイルのみを一つのフォルダへまとめてキャッシュを作成する。
```sh
python3 preprocess.py <speaker-audio-directory> --output speaker_cache
```
2. 明示的に選んだdecoderとdiscriminatorをファインチューニングする。
```sh
python3 train_dec.py --dataset-cache speaker_cache --decoder-path <decoder-checkpoint> --discriminator-path <discriminator-checkpoint> --training-state-path <training-state-checkpoint> --steps <target-total-step> --fp16
```
3. ベクトル検索用の辞書を作成する。これにより毎回音声ファイルをエンコードする必要がなくなります。
```sh
python3 extract_index.py --dataset-cache speaker_cache --output <dictionary-output>
```
4. 推論する際は`-idx <辞書ファイル>`オプションをつけることで任意の辞書データを読み込むことができます。

### 学習オプション
- `--fp16`を付けるとmixed precisionで学習する。
- `--batch-size <number>`でバッチサイズを変更する。既定値は1秒区間の`32`。
- `--steps <number>`で総学習step数を変更する。既定値は`600000`。
- `--device <device>`で演算deviceを変更する。既定値は`cuda`。
- decoder、discriminator、optimizer、scalerをatomicに保存し、同じ`--training-state-path`を指定すると同一stepから再開する。

## 推論
1. `inputs` フォルダを作成する。
2. `inputs` フォルダに変換したい音声ファイルを入れる
3. 推論スクリプトを実行する
```sh
python3 infer.py -t <ターゲットの音声ファイル>
```

### 追加のオプション
- `-a <0.0から1.0の数値>`で元音声情報の透過率を設定できます。
- `--normalize`で音量を正規化できます。
- `-d <デバイス名>` で演算デバイスを変更できます。もともと高速なのであまり意味がないかもしれませんが。
- `-p <音階>` でピッチシフトを行うことができます。男女間の音声変換に有用です。

## pyaudioによるリアルタイム推論 (テスト段階の機能です)
1. オーディオデバイスのIDを確認
```sh
python3 audio_device_list.py
```

2. 実行
```sh
python3 infer_streaming.py -i <入力デバイスID> -o <出力デバイスID> -l <ループバックデバイスID> -t <ターゲットの音声ファイル>
```
(ループバックのオプションはつけなくても動作します。)

streaming入出力は現在24kHzのaudio deviceを必要とする。異なるsample rateはchunk単位の不連続なresampleを行わず、明示的に拒否する。

## テスト
```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

## 参考文献
- [FastSVC](https://arxiv.org/abs/2011.05731)
- [kNN-VC](https://arxiv.org/abs/2305.18975)
- [WavLM](https://arxiv.org/pdf/2110.13900.pdf) (Fig. 2)
- [StreamVC](https://arxiv.org/abs/2401.03078v1)
- [Hifi-GAN](https://arxiv.org/abs/2010.05646)
- [AdaIN](https://arxiv.org/abs/1703.06868)
- [Seed-VC](https://github.com/Plachtaa/seed-vc)
- [ESTVocoder](https://arxiv.org/abs/2411.11258)
- [LLVC](https://arxiv.org/abs/2311.00873)
