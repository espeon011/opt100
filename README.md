# 「Python 言語による実務で使える 100+ の最適化問題」練習用リポジトリ

[https://scmopt.github.io/opt100](https://scmopt.github.io/opt100)

## Jupyter 起動

```
$ uv run jupyter lab --no-browser --ServerApp.ip="*" --ServerApp.custom_display_url="http://$(hostname):8888/lab"
```

## Marimo 起動

```
$ uv run marimo --development-mode edit --headless --host 0.0.0.0 --sandbox --no-token
```

Marimo をリモートマシンで起動している場合,
表示された IP アドレスをリモートマシンの IP アドレスに変更してブラウザからアクセスする. 

## Marimo notebook から Jupyter notebook への変換

```
$ uv run marimo export ipynb --include-outputs <Marimo notebook name(.py file)> --output <Jupyter notebook name(.ipynb file)>
```

## Submodule の更新

```
$ git submodule update --remote
$ git add .
$ git commit
```
