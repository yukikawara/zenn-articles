---
title: "画像から三次元形状の復元を行う NeRF のレビュー論文を読んだメモ"
emoji: "📑"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["NeRF", "3D", "CV", "ML"]
published: true
publication_name: "fusic"
---

こんにちは、初めましての方は初めまして。株式会社 [Fusic](https://fusic.co.jp/) の瓦です。夏が近づいて暑くなってきました。冷蔵庫の扉を開けるたびに感じる涼しさが心地よく、無駄に多い頻度で飲み物を冷蔵庫から取りだしては涼みつつ水分補給して、お腹を壊しています。

この記事では [Neural Radiance Fields (NeRFs): A Review and Some Recent Developments](https://arxiv.org/abs/2305.00375) という論文を読み、NeRF の関連技術について知っていこうと思います。ちなみにこの記事を書いている時点での僕自身の背景知識としては、「NeRF？ なんか写真から 3D が出来るやつだっけ？」みたいな感じです。そのため、まだちゃんとした理論を理解しているわけでもなく、関連技術についての知識についてもこの記事を書く際に勉強したくらいなので、間違っている部分があるかもしれません。そのため、おかしな部分を見つけた場合はご指摘してもらえると嬉しいです。

# 概要
NeRF は、ある物体を様々な角度から撮影した写真をもとに、その物体の三次元の形状を復元し、自由な視点からの画像を生成する技術です。テキストの説明よりも、[NeRF の元論文](https://arxiv.org/abs/2003.08934)の著者らによる以下の動画を見る方が何をやりたいのかが理解しやすいと思います。

https://youtu.be/JuH79E8rdKc

NeRF が提案されて以降、三次元形状の復元に対して様々な手法が提案されてきました。例えば、NeRF では自由視点の画像を生成するためにたくさんの画像と時間が必要になるという問題があります。この問題に対して、PixelNeRF や RegNeRF という手法が提案されました。また、実世界での NeRF の実現に向けて、Mip-NeRF や Raw NeRF、NeRF in the Wild という手法では綺麗ではない画像から 3D の復元を試みています。

# NeRF について
[このサイト](https://techblog.leapmind.io/blog/introduction-to-nerf-without-math/)がざっくりと概要をつかむために分かりやすかったです。ポイントらしい部分は

- 三次元の座標 $(x,y,z)$ とそのカメラ方向 $(\theta, \Phi)$ を入力として、色情報 $(r,g,b)$ と密度 $\sigma$ を返すようにモデルの学習を行う。
- モデルの出力を用いてレンダリングすることで三次元の形状が分かる。
- 単純に学習を行うと詳細な形状の学習（高周波情報の学習）がうまくいかないので、position encoding という手法を用いて入力を高周波領域にマッピングしている。

です。

つまりモデルへの入力は単なる画像ではなく「ある位置におけるある角度」であり、モデルの出力として「その位置からはこう見えるよ」というのが出てくるので、そのモデルを使って推論すると新たな視点から見たらどう見えるのかが分かる、ということですね。

# 関連技術
NeRF を使用することで画像から三次元形状の復元ができるようになった一方で、いくつかの課題が残されています。課題の一つとして、「入力として三次元の座標とカメラの角度」が必要になるのですが、普通の写真ではそんなものを記録していません。幸いにもこの課題に対しては、これらの値を推定する方法がいくつかあるらしく、ある程度解決できるようです。また別の課題として、「学習に使用する画像が少ないと三次元形状の復元が出来ない（NeRF では少なくとも 80 枚の画像が必要らしい）」、「鏡の反射や背景、天候など、撮る時間によって変わるものがあると三次元形状の復元に失敗する」というものがあります。この論文では、これらの問題に対して解決を図った PixelNeRF, RegNeRF, Mip-NeRF, Raw NeRF, NeRF in the Wild を紹介しています。

## 少数画像からの三次元形状復元
### PixelNeRF
[プロジェクトページのリンク](https://alexyu.net/pixelnerf/)

元の NeRF では多くの画像がないと三次元形状の復元がうまく行えません。また、元の NeRF では Multi Layer Perceptron (MLP) というモデルを使用していますが、MLP では画像を一列にして処理をするため、空間情報をうまく扱うことが出来ません。その問題に対して、PixelNeRF では Convolutional Neural Network (CNN) を用いて画像の特徴量を計算し、それを NeRF モデルへと入力することで解決を図っています。単純な物体であれば一枚の画像から、複雑な図形でも三枚程度の画像から三次元形状の復元が出来ています。

### RegNeRF
[プロジェクトページのリンク](https://m-niemeyer.github.io/regnerf/index.html)

著者らによると、元の NeRF が少数画像で三次元形状の復元をうまく出来ないのは、幾何学的な一貫性を考慮した推論が出来ずにレンダリングがうまくいかないこと、訓練開始時に発散してしまう（訓練を進めてもうまく収束しなくなる？）のが原因らしいです。そのため、この論文では（幾何的な？ あまりよく分かっていません）正則化を行い、さらに normalizing flow を用いることで一貫性が保たれるようにしているとのことです。また、CNN ではなく MLP を用いているため、PixelNeRF と比較して計算量も軽く済んでいることがメリットとして挙げられています。

## 綺麗でない画像からの三次元形状復元
### Mip-NeRF
元の NeRF やその亜種は、解像度の異なる画像を使用する場合に細かい部分がぼやけてしまい、うまくレンダリング出来ないという問題があります。これは各ピクセルごとに様々な視点からの画像を推論しレンダリングすることで解決できますが、かなりの時間がかかってしまいます。この問題に対して Mip-NeRF では、ある点ではなく円錐の断面のようなものの期待値として推論し、それをもとにレンダリングを行うことで詳細な部分の描画もできるようにしています。また、効率的にレンダリングが行えるようになるため、NeRF より高精度で高速になったと報告しています。

### Raw NeRF
[プロジェクトページのリンク](https://bmild.github.io/rawnerf/)

この論文では新たなモデルの提案というよりも、後処理などの画像処理を行うことでフォーカスや露出効果を付与する手法を提案しています。この論文名が `NeRF in the Dark` であることからも分かるように、特に暗闇における画像からの三次元形状の復元に焦点をあてています。元の NeRF では Low Dynamic Range な画像を使用しているのですが、画像の質としては低く、特に暗所部分の詳細が潰れてしまっているらしいです。そこでこのモデルを用いることで、照明が暗くてもうまく三次元形状の復元を行うことができ、さらに後処理でフォーカスなどをいじることも出来るとのことです（この後処理の部分があまり分かってません、すみません）

### NeRF in the Wild (NeRF-W)
[プロジェクトページのリンク](https://nerf-w.github.io/)

元の NeRF やその亜種は、屋内のある部屋など使用する画像に制限があり、屋外のような開けた場所での画像ではうまくいかないという問題があります。また、天候や移動する物体などの時間によって変化するオブジェクトが、おかしなオブジェクトとしてレンダリングされてしまうという問題もあります。この問題に対して、NeRF-W では時間によって変化しないオブジェクトと一時的に現れている（時間経過で変化する）オブジェクトを分離することで解決を図っています。分離したものから時間によって変化しないオブジェクトをレンダリングすることで、三次元形状の復元を行えています。プロジェクトページを見ると分かりますが、屋外の建物でもかなり綺麗に三次元形状の復元を行えていることが分かります。

# まとめ
以上、論文を読み、その内容についてまとめてみました。NeRF の問題やその関連技術についてまとめられており、（初心者意見ではありますが）概要をざっくり掴むには良さそうです。NeRF によって画像から三次元の形状を復元できるので、AR や VR と組み合わせることで色々面白いことが出来そうです（例えば VR と絡めることで、家にいながら世界遺産を訪れることが出来ます。というか個人的にあったら嬉しいなと思っています）ただ、実際に技術として存在するものの、NeRF でどのくらいのことが可能なのかは論文からは分かりづらいです。そのため、時間が取れた時に [Nerfstudio](https://docs.nerf.studio/en/latest/) という簡単に NeRF を触ることが出来るツールを使って、実際に NeRF やその亜種を触ってみようと思います。

最後に宣伝になりますが、機械学習でビジネスの成長を加速するために、[Fusic](https://fusic.co.jp/)の機械学習チームがお手伝いたします。機械学習のPoCから運用まで、すべての場面でサポートした実績があります。もし、困っている方がいましたら、ぜひ[Fusic](https://fusic.co.jp/)にご相談ください。[お問い合わせ](https://fusic.co.jp/contact/)からでも気軽にご連絡いただけます。また[TwitterのDM](https://twitter.com/kawara_fusic)からでも大歓迎です！
