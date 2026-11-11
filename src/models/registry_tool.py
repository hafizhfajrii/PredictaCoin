import argparse
import logging
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from train import DEFAULT_EXPERIMENT, default_tracking_uri, git_info, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_NAME = "predictacoin-trend-xgb"
DEFAULT_LATEST = PROJECT_ROOT / "data" / "processed" / "latest.parquet"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "models" / "production"
DEFAULT_YAML = PROJECT_ROOT / "models" / "production_model.yaml"
STAGES = ("Staging", "Production")
HYPER_KEYS = ("n_estimators", "max_depth",
              "learning_rate", "C", "class_weight")
MODEL_DESCRIPTION = (
    "Klasifikasi tren harga 1 jam ke depan (Naik / Turun / Stabil) untuk BTC dan ETH "
    "berbasis XGBoost. Input: 21 fitur teknikal per jam. Output: indeks kelas "
    "(0=Naik, 1=Turun, 2=Stabil). Dilatih pada data berversi DVC; lihat tag tiap versi."
)

logger = logging.getLogger("registry_tool")


def get_run(client, experiment_name, run_name):
    """Ambil run terbaru dengan nama `run_name` pada experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' tidak ditemukan.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise ValueError(
            f"Run '{run_name}' tidak ditemukan di '{experiment_name}'.")
    return runs[0]


def list_versions(client, name):
    """Daftar versi model terurut dari yang terlama."""
    versions = client.search_model_versions(f"name='{name}'")
    return sorted(versions, key=lambda v: int(v.version))


def alias_map(client, name):
    """Petakan nomor versi -> daftar alias (API pencarian tidak mengisi alias)."""
    mapping = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        mapping.setdefault(str(version), []).append(alias)
    return mapping


def parse_value(text):
    """Ubah string parameter MLflow menjadi int/float bila memungkinkan."""
    for cast in (int, float):
        try:
            return cast(text)
        except (TypeError, ValueError):
            continue
    return text


def cmd_register(args, client):
    """Daftarkan model dari run sebagai versi baru, lengkap dengan metadata."""
    run = get_run(client, args.experiment_name, args.run_name)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    model_uri = tags.get("model_uri")
    if not model_uri:
        raise ValueError(
            "Run tidak memiliki tag model_uri (model belum di-log).")

    for existing in list_versions(client, args.name):
        if existing.tags.get("run_id") == run.info.run_id:
            logger.warning("Run ini sudah terdaftar sebagai versi %s. Dilewati.",
                           existing.version)
            return

    version = mlflow.register_model(model_uri, args.name).version
    hyper = {k: params[k] for k in HYPER_KEYS if k in params}
    description = (
        f"{params['model']} dari run {args.run_name}. Parameter: {hyper}. "
        f"CV F1-macro {metrics['cv_f1_macro_mean']:.4f}, "
        f"uji F1-macro {metrics['test_f1_macro']:.4f}. "
        f"Data {tags.get('data_version')} (md5 {params['data_md5'][:8]}), "
        f"commit {tags.get('git_commit')}.")
    client.update_model_version(args.name, version, description=description)
    if not client.get_registered_model(args.name).description:
        client.update_registered_model(
            args.name, description=MODEL_DESCRIPTION)

    metadata = {
        "run_id": run.info.run_id, "run_name": args.run_name,
        "model_type": params["model"], "seed": params.get("seed"),
        "cv_f1_macro_mean": f"{metrics['cv_f1_macro_mean']:.4f}",
        "test_f1_macro": f"{metrics['test_f1_macro']:.4f}",
        "test_accuracy": f"{metrics['test_accuracy']:.4f}",
        "data_version": tags.get("data_version"), "data_md5": params["data_md5"],
        "git_commit": tags.get("git_commit"),
    }
    for key, value in metadata.items():
        client.set_model_version_tag(args.name, version, key, str(value))
    logger.info("Terdaftar: %s versi %s (run %s)",
                args.name, version, args.run_name)


def version_row(client, mv, aliases):
    """Ringkasan satu versi model untuk tabel perbandingan."""
    run = client.get_run(mv.run_id)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    row = {
        "stage": mv.current_stage, "alias": ",".join(aliases.get(str(mv.version), [])) or "-",
        "run": tags.get("mlflow.runName"),
        "data": f"{tags.get('data_version')} ({params['data_md5'][:8]})",
    }
    row.update({k: params.get(k, "-") for k in HYPER_KEYS})
    row.update({
        "cv_f1_macro_mean": round(metrics["cv_f1_macro_mean"], 4),
        "cv_f1_macro_std": round(metrics["cv_f1_macro_std"], 4),
        "test_f1_macro": round(metrics["test_f1_macro"], 4),
        "test_accuracy": round(metrics["test_accuracy"], 4),
        "train_f1_macro": round(metrics["train_f1_macro"], 4),
        "gap_train_test": round(metrics["train_f1_macro"] - metrics["test_f1_macro"], 4),
    })
    return row


def cmd_compare(args, client):
    """Cetak perbandingan seluruh versi dan kandidat terbaik."""
    versions = list_versions(client, args.name)
    if not versions:
        raise ValueError(f"Model '{args.name}' belum punya versi.")
    aliases = alias_map(client, args.name)
    rows = {f"v{mv.version}": version_row(
        client, mv, aliases) for mv in versions}
    table = pd.DataFrame(rows)
    print(f"\nModel: {args.name}\n")
    print(table.to_string())
    best = max(rows, key=lambda v: (rows[v]["cv_f1_macro_mean"],
                                    -rows[v]["gap_train_test"]))
    print(f"\nKandidat terbaik (cv_f1_macro_mean tertinggi, gap latih-uji "
          f"sebagai pembeda): {best}")


def cmd_promote(args, client):
    """Transisi stage (dan alias) sebuah versi: Staging atau Production."""
    stage = args.stage.capitalize()
    if stage not in STAGES:
        raise ValueError(f"Stage harus salah satu dari {STAGES}.")
    try:
        with warnings.catch_warnings():  # API stage deprecated di MLflow 3.x
            warnings.simplefilter("ignore", FutureWarning)
            client.transition_model_version_stage(
                args.name, args.version, stage,
                archive_existing_versions=(stage == "Production"))
    except Exception as exc:
        logger.warning(
            "Transisi stage gagal/tidak tersedia (%s). Alias tetap diatur.", exc)
    client.set_registered_model_alias(args.name, stage.lower(), args.version)

    if stage == "Production":  # versi yang naik ke Production meninggalkan Staging
        try:
            staged = client.get_model_version_by_alias(args.name, "staging")
            if str(staged.version) == str(args.version):
                client.delete_registered_model_alias(args.name, "staging")
        except MlflowException:
            pass
    logger.info("%s versi %s -> %s (alias '%s')", args.name, args.version,
                stage, stage.lower())
    aliases = alias_map(client, args.name)
    for mv in list_versions(client, args.name):
        print(f"  v{mv.version}: stage={mv.current_stage}, "
              f"aliases={aliases.get(str(mv.version), [])}")


def load_inference_data(path, features):
    """Muat baris baru (tanpa label) untuk uji inferensi."""
    if not path.exists():
        raise FileNotFoundError(f"Data inferensi tidak ditemukan: {path}")
    df = pd.read_parquet(path)
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(
            f"Kolom fitur hilang pada data inferensi: {sorted(missing)}")
    df = df.dropna(subset=features)
    if df.empty:
        raise ValueError(
            "Tidak ada baris dengan fitur lengkap untuk inferensi.")
    return df


def cmd_verify(args, client):
    """Muat model Production dari registry dan prediksi data baru."""
    uri = (f"models:/{args.name}/Production" if args.by_stage
           else f"models:/{args.name}@{args.alias}")
    started = time.perf_counter()
    model = mlflow.pyfunc.load_model(uri)
    load_seconds = time.perf_counter() - started

    mv = client.get_model_version_by_alias(args.name, args.alias)
    features = model.metadata.get_input_schema().input_names()
    data = load_inference_data(Path(args.data), features)

    started = time.perf_counter()
    pred = model.predict(data[features].astype("float64"))
    infer_ms = (time.perf_counter() - started) * 1000

    classes = load_config()["model"]["classes"]
    result = pd.DataFrame({
        "coin_id": data["coin_id"].values,
        "timestamp": data["timestamp"].astype(str).values,
        "prediksi": [classes[int(i)] for i in pred],
    })
    print(f"\nModel   : {uri}")
    print(f"Versi   : {mv.version} (stage {mv.current_stage})")
    print(f"Load    : {load_seconds:.2f} detik | Inferensi: {len(result)} baris "
          f"dalam {infer_ms:.1f} ms\n")
    print(result.to_string(index=False))


def cmd_export(args, client):
    """Salin model Production ke folder lokal dan tulis metadata YAML."""
    mv = client.get_model_version_by_alias(args.name, args.alias)
    run = client.get_run(mv.run_id)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{args.name}@{args.alias}", dst_path=str(out_dir))

    metadata = {
        "registered_model": {"name": args.name, "version": int(mv.version),
                             "alias": args.alias, "stage": mv.current_stage},
        "source_run": {"run_id": mv.run_id, "run_name": tags.get("mlflow.runName")},
        "selection_reason": args.reason,
        "params": {k: parse_value(params[k]) for k in HYPER_KEYS if k in params},
        "metrics": {k: round(metrics[k], 4) for k in (
            "cv_f1_macro_mean", "cv_f1_macro_std", "test_f1_macro",
            "test_accuracy", "train_f1_macro")},
        "data": {"version": tags.get("data_version"), "md5": params.get("data_md5"),
                 "rows": int(params["data_rows"]),
                 "start": params.get("data_start"), "end": params.get("data_end")},
        "code": {"git_commit": git_info()[0], "training_commit": tags.get("git_commit")},
        "artifact": {"dir": str(out_dir.relative_to(PROJECT_ROOT)),
                     "dvc_pointer": f"{out_dir.relative_to(PROJECT_ROOT)}.dvc"},
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    yaml_path = Path(args.yaml)
    yaml_path.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
                         encoding="utf-8")
    logger.info("Model versi %s -> %s", mv.version, out_dir)
    logger.info("Metadata -> %s", yaml_path)


COMMANDS = {"register": cmd_register, "compare": cmd_compare, "promote": cmd_promote,
            "verify": cmd_verify, "export": cmd_export}


def parse_args(argv=None):
    """Parse argumen CLI dengan subperintah."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--name", default=DEFAULT_MODEL_NAME,
                        help="nama model terdaftar")
    common.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    common.add_argument("--tracking-uri", default=default_tracking_uri())

    parser = argparse.ArgumentParser(
        description="Manajemen MLflow Model Registry.")
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", parents=[
                         common], help="daftarkan model dari run")
    reg.add_argument("--run-name", required=True)

    sub.add_parser("compare", parents=[common], help="bandingkan semua versi")

    pro = sub.add_parser("promote", parents=[common], help="transisi stage")
    pro.add_argument("--version", type=int, required=True)
    pro.add_argument("--stage", required=True, help="Staging atau Production")

    ver = sub.add_parser("verify", parents=[
                         common], help="uji inferensi model Production")
    ver.add_argument("--alias", default="production")
    ver.add_argument("--by-stage", action="store_true",
                     help="muat lewat URI stage (models:/nama/Production)")
    ver.add_argument("--data", default=str(DEFAULT_LATEST))

    exp = sub.add_parser("export", parents=[
                         common], help="ekspor model + metadata YAML")
    exp.add_argument("--alias", default="production")
    exp.add_argument("--out-dir", default=str(DEFAULT_EXPORT_DIR))
    exp.add_argument("--yaml", default=str(DEFAULT_YAML))
    exp.add_argument("--reason", default="(isi alasan pemilihan versi)")
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point. Mengembalikan exit code (0 = sukses, 1 = gagal)."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    mlflow.set_tracking_uri(args.tracking_uri)
    try:
        COMMANDS[args.command](args, MlflowClient())
    except (ValueError, FileNotFoundError, MlflowException) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
