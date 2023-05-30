---
title: "nerfstudioでスマホで撮影した動画から三次元データを作ってみる"
emoji: "🍰"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["NeRF", "CV", "3D"]
published: true
publication_name: fusic
---

こんにちは、初めましての方は初めまして。株式会社 [Fusic](https://fusic.co.jp/) の瓦です。梅雨に入って降り込める雨を眺めるたびに鬱々とした気持ちになる反面、「引きこもるための口実が出来て良かった」と思ってしまいます。

以前[こちらの記事](https://zenn.dev/fusic/articles/paper-reading-nerf-review)で NeRF という、画像から三次元形状を復元する技術についてのレビュー論文の紹介をしました。この記事では、NeRF を手軽に試せる [nerfstudio](https://docs.nerf.studio/en/latest/index.html) というツールを使って自分で撮影した動画から物体の三次元形状の復元を行い、さらに点群データとして出力して python でデータを触ってみたいと思います。

# nerfstudio の簡単な紹介

> Nerfstudio provides a simple API that allows for a simplified end-to-end process of creating, training, and visualizing NeRFs. The library supports an interpretable implementation of NeRFs by modularizing each component. With modular NeRF components, we hope to create a user-friendly experience in exploring the technology. Nerfstudio is a contributor-friendly repo with the goal of building a community where users can easily build upon each other’s contributions.

（[nerfstudio のページ](https://docs.nerf.studio/en/latest/index.html)より引用）

引用した文に書いてある通り、nerfstudio は NeRF を簡単に扱えるようにするためのツールです。データの前処理や訓練、結果の描画部分を分離することでそれぞれの処理を別々に行えます。チェックポイントも保存されるため、途中から再開することや、結果の確認も簡単です。また、NeRF の手法を自分で定義したものにも簡単に差し替えることが出来ます^[[https://docs.nerf.studio/en/latest/nerfology/methods/index.html#own-method-docs](https://docs.nerf.studio/en/latest/nerfology/methods/index.html#own-method-docs)]。

# 実際に試してみる

試してみた環境のスペックは以下になります。

- CPU: i7-13700K
- GPU: GeForce RTX 4070 Ti
- メモリ: 64 GB

公式の docker イメージが提供されているので、それを用いてコンテナを建てて、その中で nerfstudio を実行します。ドキュメントに使い方^[[https://docs.nerf.studio/en/latest/quickstart/installation.html#use-docker-image](https://docs.nerf.studio/en/latest/quickstart/installation.html#use-docker-image)]が載っているので、その例をほぼそのまま使用しましょう。

```bash
docker run --gpus all \
    -v /path/to/data:/workspace/ \
    -v /home/<USER_NAME>/.cache/:/home/user/.cache/ \
    --name <CONTAINER_NAME> \
    -p 7007:7007 \
    -it \
    --shm-size=12gb \
    dromni/nerfstudio:<VERSION>
```

`/path/to/data` はデータを置くフォルダに、`<CONTAINER_NAME>` はコンテナ名に、`<VERSION>` はイメージのバージョンに置き換えてください（執筆時に使用したイメージのバージョンは `0.3.1` です）ポート番号はレンダリング結果を確認するためにビューアに接続するためのものとなります。既に使用されている場合は適宜変更してください（ちなみに : の前がコンテナ外のポート番号、後ろがコンテナ内のポート番号です）

ちょうどケーキ^[KAKA というチーズケーキの専門店です。とても美味しい]が冷蔵庫にあったので、NeRF を試す対象はそのケーキにします。撮った動画データは docker コンテナの `/workspace/data/` に `cake.mp4` として置きました。ちなみに動画はスマホで撮影したもので、専門的な道具を使用したものではありません。

自身のデータを使用する方法はドキュメント^[[https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html)]にあるので、それに従っていきます。まず動画から画像をサンプリングし、COLMAP を用いてカメラの位置などの推定を行います。

```bash
ns-process-data video --data data/cake.mp4 --output-dir data/cake
```

実行して少し時間が経過した後、`data/cake` 以下に `colmap`, `images`, `images_2`, `images_4`, `images_8`, `transforms.json` が作成されます。恐らくデフォルトだと 300 枚ほどの画像が動画からサンプリングされると思います。変更したい場合は引数として `--num-frames-target <枚数>` を付与すると良いです。他の引数については `ns-process-data videos -h` を実行して適宜参照してください。

前処理が終わった後、NeRF モデルの訓練を実行します。これもドキュメントにある通りに実行します。

```bash
ns-train nerfacto --data data/cake
```

実行すると訓練が進んでいく様子が出力されると思います。レンダリング結果を見れるリンクも表示されると思うので、ブラウザで確認してみます。

https://github.com/yukikawara/zenn-articles/tree/main/movies/try-nerfstudio/cake1.gif

https://github.com/yukikawara/zenn-articles/tree/main/movies/try-nerfstudio/cake2.gif

GIF なので解像度がかなり粗いですが、ケーキの三次元形状が復元できていることが分かります。右の設定で解像度の変更も行えるので、より綺麗な画像をレンダリングしたり、さらには depth 画像の出力も行うことができ、レンダリングソフトがかなり使いやすいです。さらに EXPORT では点群とメッシュデータの出力も行えます。

![](/images/try-nerfstudio/export-setting.png)

ここでは点群データを出力させてみます。欲しい点群数、中心と欲しい範囲、出力先のディレクトリを指定することで、コマンドが生成されます。ブラウザ上からはダウンロード出来ないので、コマンドを叩く必要があります（ブラウザ上でダウンロード出来たら便利なんだけどなと思うのですが、難しいのでしょうか）また、座標の表示も出来なさそうなので、推測で記入します。右上の `Refresh Page` を押すことで (0, 0, 0) が映る視点が表示されると思うので、それを参考に推測しましょう。

コマンドを実行すると、出力先のディレクトリに `point_cloud.ply` というファイル名で点群データが出力されます。実際に出力できているのか、python コードを書いて可視化し、確かめてみます。

```python
import open3d as o3d

filename = "exports/pcd/point_cloud.ply"
pcd = o3d.io.read_point_cloud(filename)
o3d.visualization.draw_geometries([pcd])
```

可視化には [open3d](http://www.open3d.org/docs/release/index.html) というライブラリを使用しました。コードは[詳解　3次元点群処理　Pythonによる基礎アルゴリズムの実装](https://www.kspub.co.jp/book/detail/5293430.html)という本のコードを参考にしました。少し話はそれますが、この本では点群の扱い方（ファイルフォーマットなど）や点群データの特徴量の抽出方法、二つの点群データが存在する場合の位置合わせの方法など、面白いトピックがコードも交えてわかりやすく説明されているので、興味がある方はぜひ買ってみて損はないと思います。

https://github.com/yukikawara/zenn-articles/tree/main/movies/try-nerfstudio/cake_point.gif

可視化してみた結果が上の GIF になります。ちょっと余計な部分の点群もありますが、ケーキの形の点群が表示されていることが分かります。

# まとめ
この記事では nerfstudio という、NeRF を簡単に扱えるツールを試してみました。自分のスマホでさっと撮った動画から簡単に三次元形状を復元することができ、さらにそれを点群データとして出力することも出来ました。これを使用すれば、例えば unity などの 3D 開発ソフトで現実の物体を使用したいときにさっと導入することが出来そうです。また、風景を撮影し、それをもとに三次元形状の復元なども出来るのではないかと思います（遠い物体などは難しいかもしれませんが）

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
