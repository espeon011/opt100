# 「Python 言語による実務で使える 100+ の最適化問題」練習用リポジトリ

[https://scmopt.github.io/opt100](https://scmopt.github.io/opt100)

## 参考

- [ohken322/opt100](https://github.com/ohken322/opt100)
- [okaduki/opt100](https://github.com/okaduki/opt100)

## Marimo 起動

```
uv run marimo edit --headless --host 0.0.0.0 --sandbox --no-token
```

Marimo をリモートマシンで起動している場合,
表示された IP アドレスをリモートマシンの IP アドレスに変更してブラウザからアクセスする. 

## Submodule

### 初回

親リポジトリに記録されているコミットでサブモジュールを取得する.

```
git submodule update --init
```

clone 時に `git clone --recurse-submodules <URL>` としておけばこの手順は不要.

### 更新

各サブモジュールをリモートの最新コミットに更新する.

```
git submodule update --init --remote
```

更新されたサブモジュールを確認し, 親リポジトリにコミットする.
`git add .` だとサブモジュール以外の変更も含まれるので, サブモジュールのパスを指定する.

```
git submodule status
git add $(git config -f .gitmodules --get-regexp '\.path$' | awk '{print $2}')
git commit -m "submodule update"
```

### 他の環境で更新・パス変更を取り込む

pull した後, サブモジュールの URL/パス設定を同期して取得し直す.

```
git pull
git submodule sync
git submodule update --init
```
