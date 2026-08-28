// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]

use std::io::Cursor;
use std::path::PathBuf;
use std::sync::LazyLock;

use divan::Bencher;
use futures::StreamExt;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_chunks;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::DeltaScheme;
use vortex_btrblocks::schemes::integer::EliasFanoScheme;

const ARMS: [&str; 2] = ["elias-fano", "delta"];

const PARQUET: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../vortex-bench/data/polarsignals/100000/parquet/stacktraces.parquet"
);

const COLUMN: &str = "locations";
const OFFSETS: &str = "locations.elements.lines.offsets";
const READS: usize = 100_000;
const HOP: usize = 7919;

fn path_for(arm: &str) -> String {
    format!("/tmp/ef-vs-delta-random-access/{arm}.vortex")
}

fn encoding_for(arm: &str) -> &'static str {
    if arm == "elias-fano" {
        "vortex.elias_fano"
    } else {
        "fastlanes.delta"
    }
}

static RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
});

async fn write_file(arm: &str) -> u64 {
    let chunked = parquet_to_vortex_chunks(PathBuf::from(PARQUET))
        .await
        .unwrap();

    let mut banned = Vec::new();
    if arm == "elias-fano" {
        banned.push(DeltaScheme::default().id());
    } else {
        banned.push(EliasFanoScheme::default().id());
    }
    let compressor = BtrBlocksCompressorBuilder::default().exclude_schemes(banned);
    let strategy = WriteStrategyBuilder::default()
        .with_btrblocks_builder(compressor)
        .build();

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    let stream = chunked.into_array().to_array_stream();
    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut cursor, stream)
        .await
        .unwrap();

    std::fs::write(path_for(arm), &bytes).unwrap();
    u64::try_from(bytes.len()).unwrap()
}

async fn open_file(arm: &str) -> VortexFile {
    SESSION
        .open_options()
        .with_layout_reader_cache()
        .open_path(path_for(arm))
        .await
        .unwrap()
}

fn find_node(array: &ArrayRef, path: &str, wanted: &str) -> Option<ArrayRef> {
    if path == wanted {
        return Some(array.clone());
    }
    for (child, name) in array.children_iter().zip(array.children_names()) {
        let child_path = if path.is_empty() {
            name
        } else {
            format!("{path}.{name}")
        };
        if let Some(found) = find_node(child, &child_path, wanted) {
            return Some(found);
        }
    }
    None
}

async fn scan_block(file: &VortexFile) -> ArrayRef {
    let projection = select([COLUMN], root())
        .optimize_recursive(file.dtype())
        .unwrap()
        .bind(file.dtype())
        .unwrap();
    let stream = file
        .scan()
        .unwrap()
        .with_projection(projection)
        .into_array_stream()
        .unwrap();
    let mut stream = ArrayStreamExt::boxed(stream);
    let array = stream.next().await.unwrap().unwrap();
    find_node(&array, "", OFFSETS).unwrap()
}

fn probe_values(column: &ArrayRef, n: usize) {
    let mut ctx = SESSION.create_execution_ctx();
    let len = column.len();
    for i in 0..n {
        let position = (i * HOP) % len;
        let value = column.execute_scalar(position, &mut ctx).unwrap();
        divan::black_box(value);
    }
}

fn main() {
    std::fs::create_dir_all("/tmp/ef-vs-delta-random-access").unwrap();

    RUNTIME.block_on(async {
        let mut sizes: Vec<u64> = Vec::new();

        for arm in ARMS {
            let size = write_file(arm).await;
            sizes.push(size);

            let file = open_file(arm).await;
            let rows = file.row_count();
            let block = scan_block(&file).await;
            println!(
                "{arm:<11} {rows:>8} rows  {size:>10} bytes  {OFFSETS} [{}] len={}",
                block.encoding_id(),
                block.len(),
            );
            assert_eq!(block.encoding_id().to_string(), encoding_for(arm));
        }

        let ratio = (sizes[1] as f64) / (sizes[0] as f64);
        println!("delta/elias-fano size: {ratio:.5}x\n");
    });

    divan::main();
}

/// Probes against a block that is already loaded. No scan, no file IO, no task scheduling: just
/// `select1` against rebuilding a block. This is the most the encoding choice can ever be worth.
#[divan::bench(args = ARMS)]
fn probe(bencher: Bencher, arm: &str) {
    let file = RUNTIME.block_on(open_file(arm));
    let block = RUNTIME.block_on(scan_block(&file));

    bencher
        .counter(divan::counter::ItemsCount::new(READS))
        .bench(|| probe_values(&block, READS));
}

/// The same probes, but the block is scanned from the file inside the timed run. The difference
/// from `probe` is the cost of one scan.
#[divan::bench(args = ARMS)]
fn scan_then_probe(bencher: Bencher, arm: &str) {
    let file = RUNTIME.block_on(open_file(arm));

    bencher
        .counter(divan::counter::ItemsCount::new(READS))
        .bench(|| {
            let block = RUNTIME.block_on(scan_block(&file));
            probe_values(&block, READS);
        });
}
