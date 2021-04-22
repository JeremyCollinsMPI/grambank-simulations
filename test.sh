docker build -t jeremycollinsmpi/grambank .
docker run -d --rm --name grambank_simulation -e LANG="C.UTF-8" -v $PWD:/src jeremycollinsmpi/grambank python test.py