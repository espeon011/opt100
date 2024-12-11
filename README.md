# 「Python 言語による実務で使える 100+ の最適化問題」練習用リポジトリ

[https://scmopt.github.io/opt100](https://scmopt.github.io/opt100)

## Jupyter 起動

```
$ uv run jupyter lab --no-browser --ServerApp.ip="*" --ServerApp.custom_display_url="http://$(hostname):8888/lab"
```

## Marimo 起動

```
$ uv run marimo --development-mode edit --headless --sandbox --no-token
```

## Submodule の更新

```
$ git submodule update --remote
$ git add .
$ git commit
```
