

    機械学習の練習のため競争的な学習の実験: その２ 負例の研究
    (From: 2019-05-16, Time-stamp: <2019-05-15T20:23:25Z>)


** 概要

競争的学習において、勝った例を負けた側が学習するだけでなく、負けた例と
「若干」逆側になるように勝った側が学習することにより、学習が劇的に改善
した。

競争的学習とは、ここでは、単純な2入力2出力(入出力いずれもアナログ値)の
関数の学習を3層のニューラルネットワークで行う。ただし、別の初期値を持
つ同じモデル二つについて、学習の際は、正解の出力がわからないが、その二
つのモデルのうちどちらがより正解に近いか、すなわち、その「勝者」がどち
らかだけはわかるという設定とする。


** はじめに

前回の実験において、「負の学習」を試みたがうまくいかなかった。そこを何
とか改善できないかというのが今回の動機である。

ブレインストーミング的なアイデアから語りはじめよう。

負例を学習するときは、負例を選びにくくなるように空間が変形して欲しい。
「空間」が変形するというのは、予想が決まる空間の変形のことである。その
空間の中である程度ランダムに決まるのだとせねばならないのではか…。

軸(平均)の予想があって分散の予想がある。…とでもしよう。ランダムに決まっ
た予想があり、それが実際の数値と離れているかどうか…。

ランダムに決まった予想が平均から離れていたとき実際の数値が予想に近けれ
ば、平均を移動すればよいのだろうか…。

でも、負例の学習をしたいなら、以前の負例の状況が活かされないといけない
から、平均や分散の変形だけではダメになるんじゃないか？ 負例のいくつか
の平均をとったら、それは一つの負例と同じということになりかねない。

まぁ、しかし考えてみよう。負例が軸そのものと重なったなら、軸を大きく動
かさねばならないだろう。同時に分散も大きくする。逆に、負例が軸から離れ
ていれば、軸の移動はあまり考慮する必要はない。分散は軸の周りに近付けれ
ばいいのか？


いや、この軸の移動だけ取り出して、ランダムじゃないものに活かせないか？ 
つまり、負例が近くにいるとき正例を遠くに動かし、遠くにいるときは近くが
正しいとして学習してはどうだろう？

競争的学習について考える。

負け例が勝ち例と遠いなら、- α * (負け例 - 勝ち例) + 勝ち例 の α が非
常に小さいものを学習のためのデータとする。

逆に負け例が勝ち例と近いなら、- α * (負け例 - 勝ち例) + 勝ち例 の α
が大きくてよい。ただし、負け例と勝ち例が非常に近くても、α は 1 程度で
十分であろう。もちろん、そのとき α * (負け例 - 勝ち例) 〜 0 である。

分岐点を C とし、そのとき α = 1/2 になればよい。

負け例を Y、勝ち例を Y'、学習すべき例を Y'' とする。
y' = Y'' - Y', y = Y - Y' とする。

|y'| を縦軸に |y| を横軸にしたグラフを書くと、原点 0 をスタートし、
C で最大値をとり、|y| が増えるにつれて減少しながら
なだらかに 0 に近づくグラフになる。

|y'|/|y| を縦軸に |y| を横軸にしたグラフを書くと、
|y| = 0 で |y'|/|y| = 1、|y| = C で |y'|/|y| = 1/2、
あとは |y| が増えるについれて減少しながらなだらかに 0 に近づくグラフになる。

|y'|/|y| を縦軸に |y| を横軸にしたグラフは正規分布に似ている。

ゆえに |y'|/|y| = exp(-|y|**2/(2*σ**2)) とする。

|y'| = |y| * exp(-|y|**2/(2*σ**2)) となるが、
これはレイリー分布(Reyleigh distribution)といって、
上の要求するグラフに近くなる。

ただし、C のとき  |y'|/|y| = 1/2 で |y'| が最大値になるわけではない。
が、C = σ として |y'|/|y| = 0.6... で、C = σ ** 2 で最大値になる。

負の学習においては、勝った側が、

Y'' = - (Y - Y') * exp(- |Y - Y'| ** 2 / (2 * σ ** 2)) + Y' ... (3)

を学習すれば良いのではないか？

もちろん、正の学習として、負けた側が勝ち例を学ぶこともする。


…いや、これで負の学習だけをした場合、勝ち例と負け例がだんだん離れてい
くことになりそうだ。そして、真ん中に真の値を挟むような形になる…と。

むしろ、内側に進むべきではないか。

Y'' = + (Y - Y') * exp(- |Y - Y'| ** 2 / (2 * σ ** 2)) + Y'

…という感じか？

でも、もっと思い切って、勝ち例と負け例の真ん中を学習していってはどうだ
ろう？

Y'' = (1/2) * (Y - Y') + Y'

ここで (1/2) は (1/3) や (2/5) などでも良い。

上で考えたことも活かしたい。ハイブリッドに

Y'' = (1/2) * (Y - Y') - (Y - Y') * exp(- |Y - Y'| ** 2 / (2 * σ ** 2)) + Y'

または

Y'' = + (1/2) * (Y - Y') * exp(- |Y - Y'| ** 2 / (2 * σ ** 2)) + Y'

…も意味があるだろうか？


…いや、ダメだ。勝ち例と負け例の間に真の値が常にあるなら、これでよい。
しかし、そうではないことがありうる。勝ち例の外側に真の値がある場合だ。
この場合、負例の学習だけやっていると勝ち例が負け例の側に寄っていくが、
それはニセの極限で、真の値からは外れた値になりうる。

戻って、(3) であれば、どんどん外側に離れていくが、間に真の値が必ず現れ
ることになるだろう。これを活かせないか？ 勝ち例だけでなく負け例も動か
して、…というのは、すでに正例の学習でやっているな…。

正例の学習を同時にやるなら (3) は意味があるのではないか。より外側も探
索することになるから。

どうだろう？ 実験してみないことにはなんとも…。


勝ち例と負け例が一致するからと言って、真の値と等しいとは限らないという
のは (3) でも同じである。よって、Y'' は、一定値、ランダムにずれるよう
にし、学習が進むにつれてその一定値を小さくすればいいのではないか？


…といった感じでブレインストーミングした結果、とにかく (3) を調べ、そ
の後、他のアイデアも試してみることにした。


** ソース

TensorFlow を使わないバージョンの comp_learn_3.py のソースを途中から切
り出すとだいたい次のようになる。

<source>
total_loss1 = 0
total_loss2 = 0
loss_count = 0
loss_list1 = []
loss_list2 = []

for epoch in range(max_epoch):
    for iters in range(max_iters):
        batch_x = np.random.uniform(-1.0, 1.0, (batch_size, input_size))
        batch_t = answer_of_input(batch_x)

        p1 = model1.predict(batch_x)
        p2 = model2.predict(batch_x)
        d = np.sum(np.square(p1 - batch_t), axis=1, keepdims=True) \
            < np.sum(np.square(p2 - batch_t), axis=1, keepdims=True)
        t1 = np.where(d, p1, p2)

        ploss1 = model1.calc_loss(p1, t1)
        Ploss2 = model2.calc_loss(p2, t1)
        model1.backward()
        model2.backward()
        optimizer.update(model1.params, model1.grads)
        optimizer.update(model2.params, model2.grads)

        pnd = neg_coeff * (p1 - p2) \
            * np.exp(- np.sum(np.square(p1 - p2), axis=1, keepdims=True)
                     / (2 * neg_sigma ** 2))
        pn1 = pnd + p2
        pn2 = - pnd + p1
        t2_1 = np.where(~d, p1, pn2)
        t2_2 = np.where(~d, pn1, p2)

        nloss1 = model1.calc_loss(p1, t2_1)
        nloss2 = model2.calc_loss(p2, t2_2)
        model1.backward()
        model2.backward()
        noptimizer.update(model1.params, model1.grads)
        noptimizer.update(model2.params, model2.grads)
        
        loss1 = model1.calc_loss(p1, batch_t)
        loss2 = model2.calc_loss(p2, batch_t)

        total_loss1 += loss1
        total_loss2 += loss2
        loss_count += 1

        if (iters + 1) % 10 == 0:
            avg_loss1 = total_loss1 / loss_count
            avg_loss2 = total_loss2 / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f, %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss1, avg_loss2))
            loss_list1.append(avg_loss1)
            loss_list2.append(avg_loss2)
            total_loss1, total_loss2, loss_count = 0, 0, 0
</source>

上の (3) にするには neg_coeff = -1.0 にする。neg_sigma = 1.0 ではじめ
は試した。そこを計算するところなどが、comp_learn_1.py から変わっている。
あと、comp_learn_1.py は「負の学習」は negative_learning_rate がマイナ
スになるようにしていたが、今回は、あくまでもそれを正しいものとして学習
するのでプラスのままで扱っているのが違う。

これに、さらに中間の値を取る(neg_mid で中間の位置を指定)コードと、ラン
ダムな半球面上の点を足す(neg_bs_first, neg_bs_mid, neg_bs_last)コード
を足したのが comp_learn_4.py である。t2_1 と t2_2 の計算の部分だけが上
と違うのでその部分のみ下に示す。

<source>
        bs = None
        if epoch < max_epoch / 2:
            bs = neg_bs_first + (neg_bs_mid - neg_bs_first) * epoch \
                / (max_epoch / 2)
        else:
            bs = neg_bs_mid + (neg_bs_last - neg_bs_mid) * \
                (epoch - max_epoch / 2) / (max_epoch / 2)
        bs = bs * np.array([bsphere_rand(input_size) \
                            for i in range(batch_size)])

        pnd = neg_mid * (p1 - p2) + neg_coeff * (p1 - p2) * \
            np.exp(- np.sum(np.square(p1 - p2), axis=1, keepdims=True)
                   / (2 * neg_sigma ** 2)) + bs
        pn1 = pnd + p2
        pn2 = - pnd + p1
        t2_1 = np.where(~d, p1, pn2)
        t2_2 = np.where(~d, pn1, p2)
</source>

bsphere_rand(input_size) は input_size 次元の単位球面上のランダムな点
を生成する関数である。

TensorFlow バージョンはアーカイブには入っているが、ここでは割愛する。


** 実験

詳しくは前回を見ていただきたいが、{{--hidden-size=10}} より大きくない
とうまくいかなかった。{{--hidden-size=10}} の結果 comp_learn_1_2.png
を再掲する。

comp_learn_3.py を {{--neg-coeff=-1.0}}, {{--neg-sigma=1.0}} で実行す
ると、劇的に改善され、これまで以上にうまくいく。

<source>
% python comp_learn_3.py --hidden-size=10
:
(中略)
:
| epoch 300 | iter 80 / 100 | loss 0.01, 0.01
| epoch 300 | iter 90 / 100 | loss 0.01, 0.01
| epoch 300 | iter 100 / 100 | loss 0.01, 0.01
</source>

そして comp_learn_3_2.png が表示される。

前回はさほどよくなかった {{--hidden-size=7}} でさえうまくいく。
<source>
% python comp_learn_3.py
:
(中略)
:
| epoch 300 | iter 80 / 100 | loss 0.01, 0.01
| epoch 300 | iter 90 / 100 | loss 0.01, 0.01
| epoch 300 | iter 100 / 100 | loss 0.01, 0.01
</source>

そして comp_learn_3_3.png が表示される。


一方、結果は載せないが、{{--neg-coeff=1.0}} や {{--neg-coeff=0.5}} な
どはまったくうまくいかない。

{{--neg-sigma=0.5}} など小さくしたほうが最終的な学習結果はよくなるが、
学習がこころなしか遅くなるようだ。

一方、{{--learning_rate=0.0}} にして正の学習をなしにしてみると、予想通
り学習が進まないが、進まないというだけで発散するほど大きく悪化するわけ
ではないのが他のパラメータと違うところかもしれない。

comp_learn_4.py に移ろう。{{--neg-mid=0.5}}, {{--neg-bs-first=0.1}},
{{--neg-bs-mid=0.01}}, {{--neg-bs-last=0.0}} で試す。

<source>
% python comp_learn_4.py
:
(中略)
:
| epoch 300 | iter 80 / 100 | loss 0.00, 0.00
| epoch 300 | iter 90 / 100 | loss 0.00, 0.00
| epoch 300 | iter 100 / 100 | loss 0.00, 0.00
</source>

そして comp_learn_4_1.png が表示される。少し改善されるようだ。

が、そもそも {{--neg-coeff}} がないとうまくいかない。

<source>
% python comp_learn_4.py --neg-coeff=0.0 --hidden-size=10
:
(中略)
:
| epoch 300 | iter 80 / 100 | loss 0.07, 0.07
| epoch 300 | iter 90 / 100 | loss 0.06, 0.06
| epoch 300 | iter 100 / 100 | loss 0.06, 0.06
</source>

そして comp_learn_4_2.png が表示される。最初に再掲したものと比べると少
し悪化しているぐらいかもしれない。

{{--neg-mid}} だけを削ることもやってみたが、comp_learn_3.py と同じレベ
ルに落ち着き、あったほうがこころなしか良いという結果になった。


** 結論

負の学習をうまくする方法を見つけたようである。うれしい。

が、なぜそうなのか、今一つ理論的にはっきりしない。上の「はじめに」での
考察はあくまでブレインストーミングレベルのもので、大して意味はない。正
規分布のような関数を使ったが、「若干逆側になるようにする」程度の意味し
かないかもしれない。その辺りは今後の課題としたい。

また、comp_learn_4.py で結果が改善したのも謎であり、そこのところも今後
の課題となる。


** 参考

Python 関連のサイト… Numpy や TensorFlow のサイトにいろいろお世話になっ
たが、それらについては感謝の上で割愛する。前回載せた部分についても割愛
する。

  * 《機械学習の練習のため競争的な学習の実験をしてみた》。前回の実験。
    00_README.txt


** 著者

JRF ( http://jrf.cocolog-nifty.com/software/ )


** ライセンス

私が作った部分に関してはパブリックドメイン。 (数式のような小さなプログ
ラムなので。)

自由に改変・公開してください。

ちなみに『ゼロから作る Deep Learning』のソースは MIT License で、
『scikit-learn と TensorFlow による実践機械学習』のソースは Apache
License 2.0 で公開されています。その辺りをどう考えるかは読者にまかせま
す。



(This document is written in Japanese/UTF-8.)
