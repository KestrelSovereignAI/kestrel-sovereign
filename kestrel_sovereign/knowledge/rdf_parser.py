"""Maintained, bounded offline RDF 1.1 N-Triples parser adapter.

This module is the serialized-input edge for
:mod:`kestrel_sovereign.knowledge.rdf_codec`.  It deliberately exposes one
fixed parser profile: RDFLib parses caller-supplied bytes as N-Triples 1.1.
N-Triples has neither remote contexts nor import directives.  The parser runs
in a supervised worker over an in-memory byte stream, so the parent can
enforce the time budget even if a parser stalls.  Its streaming sink checks the
statement limit before retaining each triple; an over-budget input is never
first materialized as an unrestricted RDFLib graph.  The adapter does not
provide a format switch, base URI, loader, or parser-hook configuration to
application callers.
"""

from __future__ import annotations

from io import BytesIO
import multiprocessing
from multiprocessing.connection import Connection
import time
from typing import Literal

from rdflib import BNode, Literal as RdfLibLiteral, URIRef
from rdflib.exceptions import ParserError
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser

from .assertion import RDF_LANG_STRING, XSD_STRING
from .rdf_codec import (
    RdfBlankNode,
    RdfCodecError,
    RdfDataset,
    RdfImportBudgetError,
    RdfImportDocument,
    RdfImportLimits,
    RdfIri,
    RdfLiteral,
    RdfTerm,
    RdfTriple,
)


class RdfLibNTriplesParser:
    """Parse raw RDF 1.1 N-Triples bytes with a fixed offline configuration."""

    def parse(self, payload: bytes, *, limits: RdfImportLimits) -> RdfImportDocument:
        """Return an internal document after supervised streaming parsing.

        The worker receives only an in-memory byte stream and the fixed ``nt``
        parser.  It is never given a URL, file path, base URI, JSON-LD context,
        or an ``owl:imports`` resolver.  N-Triples terms may still contain
        arbitrary IRIs as inert data; the codec validates them after this
        parser boundary without dereferencing any of them.
        """
        if not isinstance(payload, bytes):
            raise RdfCodecError("serialized RDF import requires raw bytes")
        if len(payload) > limits.max_bytes:
            raise RdfImportBudgetError("RDF import exceeds the byte budget")

        started = time.monotonic()
        context = _parser_context()
        receive, send = context.Pipe(duplex=False)
        worker = context.Process(
            target=_parse_ntriples_worker,
            args=(send, payload, limits),
            daemon=True,
        )
        worker_started = False
        try:
            worker.start()
            worker_started = True
            send.close()
            result = _receive_worker_result(receive, worker, started, limits)
        finally:
            send.close()
            receive.close()
            if worker_started:
                if worker.is_alive():
                    _stop_worker(worker)
                else:
                    worker.join()

        status, detail, triples = result
        if status == "budget":
            raise RdfImportBudgetError(detail)
        if status == "invalid":
            raise RdfCodecError("invalid RDF 1.1 N-Triples input")
        if status != "ok":  # pragma: no cover - constrained by worker protocol.
            raise RdfCodecError("RDF parser worker returned an invalid result")

        return RdfImportDocument(
            dataset=RdfDataset(triples),
            received_bytes=len(payload),
            parse_seconds=time.monotonic() - started,
        )


_WorkerStatus = Literal["ok", "budget", "invalid"]
_WorkerResult = tuple[_WorkerStatus, str, tuple[RdfTriple, ...]]
_POLL_INTERVAL_SECONDS = 0.01
_TERMINATION_GRACE_SECONDS = 0.1


def _parser_context() -> multiprocessing.context.BaseContext:
    """Return one portable isolated-process context for parser supervision."""
    # ``spawn`` avoids inheriting open descriptors, event loops, or parser
    # state from a long-running server process.  It also gives Windows the
    # same isolation contract as Unix hosts.
    return multiprocessing.get_context("spawn")


def _receive_worker_result(
    receive: Connection,
    worker: multiprocessing.Process,
    started: float,
    limits: RdfImportLimits,
) -> _WorkerResult:
    """Receive one worker result or terminate it at the import deadline."""
    while True:
        elapsed = time.monotonic() - started
        remaining = limits.max_parse_seconds - elapsed
        if remaining <= 0:
            _stop_worker(worker)
            raise RdfImportBudgetError("RDF parser exceeded the time budget")
        if receive.poll(min(remaining, _POLL_INTERVAL_SECONDS)):
            try:
                result = receive.recv()
            except EOFError as error:
                raise RdfCodecError("RDF parser worker exited before returning a result") from error
            if (
                not isinstance(result, tuple)
                or len(result) != 3
                or result[0] not in {"ok", "budget", "invalid"}
                or not isinstance(result[1], str)
                or not isinstance(result[2], tuple)
            ):
                raise RdfCodecError("RDF parser worker returned an invalid result")
            return result
        if not worker.is_alive():
            raise RdfCodecError("RDF parser worker exited before returning a result")


def _stop_worker(worker: multiprocessing.Process) -> None:
    """Terminate and reap a parser worker that outlived its allowed runtime."""
    worker.terminate()
    worker.join(_TERMINATION_GRACE_SECONDS)
    if worker.is_alive():
        worker.kill()
        worker.join()


def _parse_ntriples_worker(
    send: Connection,
    payload: bytes,
    limits: RdfImportLimits,
) -> None:
    """Parse one payload without allowing a worker failure to escape its IPC protocol."""
    started = time.monotonic()
    try:
        sink = _BudgetedNTriplesSink(limits, started)
        W3CNTriplesParser(sink=sink).parse(BytesIO(payload))
        sink.assert_time_budget()
    except RdfImportBudgetError as error:
        result: _WorkerResult = ("budget", str(error), ())
    except (ParserError, UnicodeDecodeError, ValueError, RdfCodecError):
        result = ("invalid", "invalid RDF 1.1 N-Triples input", ())
    else:
        result = ("ok", "", tuple(sink.triples))
    try:
        send.send(result)
    finally:
        send.close()


class _BudgetedNTriplesSink:
    """RDFLib's streaming sink that enforces budgets before retaining a triple."""

    def __init__(self, limits: RdfImportLimits, started: float) -> None:
        self._limits = limits
        self._started = started
        self.triples: list[RdfTriple] = []

    def triple(self, subject: object, predicate: object, object_: object) -> None:
        self.assert_time_budget()
        if len(self.triples) >= self._limits.max_statements:
            raise RdfImportBudgetError("RDF import exceeds the statement-count budget")
        self.triples.append(
            RdfTriple(
                _to_term(subject),
                _to_predicate(predicate),
                _to_term(object_),
            )
        )
        self.assert_time_budget()

    def assert_time_budget(self) -> None:
        if time.monotonic() - self._started > self._limits.max_parse_seconds:
            raise RdfImportBudgetError("RDF parser exceeded the time budget")


def _to_predicate(term: object) -> RdfIri:
    if not isinstance(term, URIRef):
        raise RdfCodecError("RDF N-Triples predicate must be an IRI")
    return RdfIri(str(term))


def _to_term(term: object) -> RdfTerm:
    if isinstance(term, URIRef):
        return RdfIri(str(term))
    if isinstance(term, BNode):
        return RdfBlankNode(str(term))
    if isinstance(term, RdfLibLiteral):
        language = term.language
        datatype = (
            RDF_LANG_STRING
            if language is not None
            else str(term.datatype) if term.datatype is not None else XSD_STRING
        )
        return RdfLiteral(str(term), datatype, language=language)
    raise RdfCodecError(f"RDF N-Triples parser emitted an unsupported term {type(term).__name__}")


__all__ = ["RdfLibNTriplesParser"]
