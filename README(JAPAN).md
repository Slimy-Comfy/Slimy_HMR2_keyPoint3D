# Slimy_HMR2_keyPoint3D

HMR2.0ベースの3Dキーポイント推定ノードです。  
**WSL2・Ubuntu不要。Windows ComfyUI Portableで直接動作します。**

hmr2はノード内にローカルコピーを同梱しているため、他のノードへの影響はありません。

---

## インストール

### 1. ノードの配置

このフォルダを `ComfyUI/custom_nodes/` に配置します。

### 2. 依存パッケージのインストール

PowerShellで以下を実行します（パスは環境に合わせてください）：

```powershell
D:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\Slimy_HMR2_keyPoint3D\requirements.txt
```

### 3. 初回実行（自動ダウンロード）

ComfyUIを起動してノードを実行すると、約2.5GBのモデルデータが自動でダウンロードされ、以下のように配置されます：

```
ComfyUI/models/hmr2/
├── data/
│   ├── smpl/               ← この中にSMPL_NEUTRAL.pklを置く（手順4）
│   ├── smpl_mean_params.npz
│   └── SMPL_to_J19.pkl
└── logs/train/multiruns/hmr2/0/
    ├── checkpoints/
    │   └── epoch=35-step=1000000.ckpt
    └── model_config.yaml
```

ダウンロード完了後、`SMPL_NEUTRAL.pkl` がないためエラーが出ます。手順4へ進んでください。

### 4. SMPLモデルの配置（必須・手動）

SMPLモデルは学術ライセンスのため自己取得が必要です。

1. https://smpl.is.tue.mpg.de/ にアクセスして登録・ログイン
2. `SMPL for Python` をダウンロード
3. zipの中にある `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` を取り出す
4. ファイル名を `SMPL_NEUTRAL.pkl` に変更する
5. 以下のパスに配置：

```
ComfyUI/models/hmr2/data/smpl/SMPL_NEUTRAL.pkl
```

6. ComfyUIを再起動して再度実行してください。

---

## ノードの使い方

| 入力 | 説明 |
|------|------|
| `image` | 推論対象の画像 |
| `person_index` | `0`=全員、`1`=1人目、`1,2`=複数指定 |
| `output_filename` | 出力JSONのファイル名プレフィックス |

| 出力 | 説明 |
|------|------|
| `pose_image` | 骨格オーバーレイ画像 |
| `skeleton_only` | 黒背景の骨格のみ画像 |
| `pose_json` | 3Dキーポイントデータ（JSON） |

ノード内の「💾 JSONを保存」ボタンでJSONをブラウザからダウンロードできます。

---

## 注意事項

- 初回実行時はモデルのダウンロード・ロードに数分かかります
- NVIDIA GPU必須
- 初回実行時にインターネット接続が必要
