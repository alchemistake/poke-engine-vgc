.PHONY: release

dev:
	virtualenv -p python3 venv
	. venv/bin/activate && pip install -r poke-engine-py/requirements.txt && pip install -r poke-engine-py/requirements-dev.txt && cd poke-engine-py && maturin develop --features="poke-engine/gen4"

# Python bindings with the doubles engine (gen9championsvgc2026regmc)
dev_doubles:
	virtualenv -p python3 venv
	. venv/bin/activate && pip install -r poke-engine-py/requirements.txt && pip install -r poke-engine-py/requirements-dev.txt && cd poke-engine-py && maturin develop --no-default-features --features="champions"

upload_python_bindings:
	cd poke-engine-py && ./build_and_publish

upload_rust_lib:
	cargo publish --features "gen4"

release:
	./release

fmt:
	cargo fmt
	ruff format poke-engine-py

gen1:
	cargo build --release --features gen1 --no-default-features

gen2:
	cargo build --release --features gen2 --no-default-features

gen3:
	cargo build --release --features gen3 --no-default-features

gen4:
	cargo build --release --features gen4 --no-default-features

gen5:
	cargo build --release --features gen5 --no-default-features

gen6:
	cargo build --release --features gen6 --no-default-features

gen7:
	cargo build --release --features gen7 --no-default-features

gen8:
	cargo build --release --features gen8 --no-default-features

gen9:
	cargo build --release --features gen9 --no-default-features

# Default champions build: profile-guided (see bench/pgo.sh), output in
# target/pgo/release/. Needs `rustup component add llvm-tools-preview`.
# `make champions_plain` is the ordinary release build.
champions: champions_pgo

champions_plain:
	cargo build --release --features champions --no-default-features

# Profile-guided build of the champions binaries for THIS machine (~10% faster
# on bench/games.json; trains on the benchmark games). Output: target/pgo/release/.
# target-cpu=native and panic=abort were measured and gave no gain.
champions_pgo:
	bench/pgo.sh

# The 100-game speed test (PGO build), best of 3 x 30 runs, in microseconds.
bench: champions_pgo
	for i in 1 2 3; do target/pgo/release/bench-doubles bench/games.json 30 --sim 2>/dev/null; done | sort -n | head -1

# Same with the plain release build.
bench_plain:
	cargo build -q --release --features champions --no-default-features --bin bench-doubles
	for i in 1 2 3; do target/release/bench-doubles bench/games.json 30 --sim 2>/dev/null; done | sort -n | head -1

bench_pgo: bench

bss:
	cargo build --release --features bss --no-default-features

tera:
	cargo build --release --features gen9,terastallization --no-default-features

pytest:
	. venv/bin/activate && pytest --rootdir=poke-engine-py/python poke-engine-py/python/tests

test: pytest
	cargo test --no-default-features --features "champions"
	cargo test --no-default-features --features "terastallization"
	cargo test --no-default-features --features "gen9"
	cargo test --no-default-features --features "gen8"
	cargo test --no-default-features --features "gen7"
	cargo test --no-default-features --features "gen6"
	cargo test --no-default-features --features "gen5"
	cargo test --no-default-features --features "gen4"
	cargo test --no-default-features --features "gen3"
	cargo test --no-default-features --features "gen2"
	cargo test --no-default-features --features "gen1"

install_ci:
	pip install -r poke-engine-py/requirements.txt
	pip install -r poke-engine-py/requirements-dev.txt
	cd poke-engine-py && maturin develop --features="poke-engine/gen4"

fmt_ci:
	cargo fmt -- --check
	ruff format --check poke-engine-py

test_ci:
	pytest --rootdir=poke-engine-py/python poke-engine-py/python/tests
	cargo test --no-default-features --features "champions"
	cargo test --no-default-features --features "gen9"
	cargo test --no-default-features --features "gen8"
	cargo test --no-default-features --features "gen7"
	cargo test --no-default-features --features "gen6"
	cargo test --no-default-features --features "gen5"
	cargo test --no-default-features --features "gen4"
	cargo test --no-default-features --features "gen3"
	cargo test --no-default-features --features "gen2"
	cargo test --no-default-features --features "gen1"

ci: install_ci fmt_ci test_ci
