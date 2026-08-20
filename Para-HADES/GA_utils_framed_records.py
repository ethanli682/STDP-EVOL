import os
import pickle
import struct
from typing import Any, Dict, Iterator, Optional

import numpy as np
import zstandard


MAGIC = b"PHR1"
HEADER_STRUCT = struct.Struct("<4sII")
INDEX_STRUCT = struct.Struct("<QII")


def _lazy_import_msgpack():
    import msgpack  # type: ignore

    return msgpack


def _lazy_import_msgspec():
    import msgspec  # type: ignore

    return msgspec


def framed_paths_from_csv(csv_path: str):
    base = csv_path[:-8] if csv_path.endswith(".log.csv") else os.path.splitext(csv_path)[0]
    return base + ".log.framed.bin", base + ".log.framed.idx"


def normalize_gene_to_float32_array(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    gene = out.get("gene", None)
    if gene is not None:
        if isinstance(gene, np.ndarray):
            out["gene"] = np.asarray(gene, dtype=np.float32)
        else:
            out["gene"] = np.asarray(gene, dtype=np.float32)
    return out


def _pack_record_for_non_pickle(record: Dict[str, Any]) -> Dict[str, Any]:
    packed = dict(record)
    gene = packed.get("gene", None)
    if isinstance(gene, np.ndarray):
        gene_f32 = np.asarray(gene, dtype=np.float32)
        packed["__gene_meta__"] = {
            "shape": list(gene_f32.shape),
            "dtype": "float32",
        }
        packed["gene"] = gene_f32.tobytes(order="C")
    return packed


def _unpack_record_from_non_pickle(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("__gene_meta__", None)
    if meta and "gene" in record:
        try:
            shape = tuple(meta.get("shape", []))
            gene = np.frombuffer(record["gene"], dtype=np.float32)
            if len(shape) > 0:
                gene = gene.reshape(shape)
            record["gene"] = gene
        except Exception:
            pass
        record.pop("__gene_meta__", None)
    return record


class FramedRecordCodec:
    def __init__(self, codec: str = "pickle"):
        self.codec = codec
        if self.codec not in {"pickle", "msgpack", "msgspec"}:
            raise ValueError(f"Unsupported codec: {codec}")

    def dumps(self, record: Dict[str, Any]) -> bytes:
        normalized = normalize_gene_to_float32_array(record)
        if self.codec == "pickle":
            return pickle.dumps(normalized, protocol=pickle.HIGHEST_PROTOCOL)

        payload = _pack_record_for_non_pickle(normalized)
        if self.codec == "msgpack":
            msgpack = _lazy_import_msgpack()
            return msgpack.packb(payload, use_bin_type=True)

        msgspec = _lazy_import_msgspec()
        return msgspec.msgpack.encode(payload)

    def loads(self, data: bytes) -> Dict[str, Any]:
        if self.codec == "pickle":
            rec = pickle.loads(data)
            return normalize_gene_to_float32_array(rec)

        if self.codec == "msgpack":
            msgpack = _lazy_import_msgpack()
            rec = msgpack.unpackb(data, raw=False)
        else:
            msgspec = _lazy_import_msgspec()
            rec = msgspec.msgpack.decode(data)

        rec = _unpack_record_from_non_pickle(rec)
        return normalize_gene_to_float32_array(rec)


class FramedRecordWriter:
    def __init__(
        self,
        data_path: str,
        index_path: str,
        codec: str = "pickle",
        compression_level: int = 3,
    ):
        os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

        self.data_path = data_path
        self.index_path = index_path
        self.codec = FramedRecordCodec(codec=codec)
        self.compressor = zstandard.ZstdCompressor(level=compression_level)
        self._data = open(self.data_path, "ab")
        self._index = open(self.index_path, "ab")

    def append(self, record: Dict[str, Any]):
        payload = self.codec.dumps(record)
        compressed = self.compressor.compress(payload)

        offset = self._data.tell()
        header = HEADER_STRUCT.pack(MAGIC, len(compressed), len(payload))
        self._data.write(header)
        self._data.write(compressed)
        self._index.write(INDEX_STRUCT.pack(offset, len(compressed), len(payload)))

    def flush(self):
        self._data.flush()
        self._index.flush()

    def close(self):
        try:
            self.flush()
        finally:
            self._data.close()
            self._index.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class FramedRecordReader:
    def __init__(
        self,
        data_path: str,
        index_path: str,
        codec: str = "pickle",
    ):
        self.data_path = data_path
        self.index_path = index_path
        self.codec = FramedRecordCodec(codec=codec)
        self.decompressor = zstandard.ZstdDecompressor()
        self._data = open(self.data_path, "rb")
        self._index = open(self.index_path, "rb")
        self._entry_count = os.path.getsize(self.index_path) // INDEX_STRUCT.size

    @classmethod
    def from_csv_path(cls, csv_path: str, codec: str = "pickle"):
        data_path, index_path = framed_paths_from_csv(csv_path)
        return cls(data_path=data_path, index_path=index_path, codec=codec)

    @classmethod
    def exists_for_csv(cls, csv_path: str):
        data_path, index_path = framed_paths_from_csv(csv_path)
        return os.path.exists(data_path) and os.path.exists(index_path)

    def __len__(self):
        return self._entry_count

    def _read_index_entry(self, index: int):
        if index < 0 or index >= self._entry_count:
            raise IndexError(index)
        self._index.seek(index * INDEX_STRUCT.size)
        return INDEX_STRUCT.unpack(self._index.read(INDEX_STRUCT.size))

    def read(self, index: int) -> Dict[str, Any]:
        offset, compressed_len, raw_len = self._read_index_entry(index)
        self._data.seek(offset)
        magic, frame_compressed_len, frame_raw_len = HEADER_STRUCT.unpack(self._data.read(HEADER_STRUCT.size))
        if magic != MAGIC:
            raise ValueError("Corrupted frame magic")
        if frame_compressed_len != compressed_len or frame_raw_len != raw_len:
            raise ValueError("Corrupted frame/index length mismatch")

        compressed = self._data.read(compressed_len)
        payload = self.decompressor.decompress(compressed, max_output_size=raw_len)
        return self.codec.loads(payload)

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        for idx in range(self._entry_count):
            yield self.read(idx)

    def close(self):
        self._data.close()
        self._index.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
