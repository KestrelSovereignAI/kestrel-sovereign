/** Public startup cancellation is not a native acquisition completion receipt.
 * Keep the original operation rooted: permission/worklet loading may finish
 * late and MUST still run the client's original late-acquisition cleanup.
 */
export function createStartupSettlement() {
  let nativeStartup = null;
  let nativeSettlement = null;
  let closedError = null;
  let isClosed = false;
  let rejectClosed;
  const closed = new Promise((resolve, reject) => { rejectClosed = reject; });
  closed.catch(() => {});
  return {
    run(operation) {
      if (isClosed) return Promise.reject(closedError);
      if (nativeStartup) return Promise.reject(new Error('Voice client already started.'));
      nativeStartup = Promise.resolve().then(operation);
      const publicStartup = Promise.race([nativeStartup, closed]);
      nativeSettlement = Promise.allSettled([nativeStartup, publicStartup]).then(([native, outcome]) => [{
        ...native,
        // The native status/reason stays factual. Closing AudioContext can
        // reject a pending worklet with a DIFFERENT AbortError; keep the
        // original public withdrawal too, without mutating either error.
        // A later close observer is not evidence that close won this race.
        publicReason: outcome.status === 'rejected' ? outcome.reason : null,
      }]);
      // This settles the PUBLIC caller promptly. It neither cancels nor
      // declares completion of the original retained native operation.
      return publicStartup;
    },
    whenSettled() { return nativeSettlement || Promise.resolve([]); },
    close(error = new Error('Original voice startup was closed.')) {
      if (isClosed) return closedError;
      isClosed = true;
      closedError = error;
      rejectClosed(closedError);
      return closedError;
    },
  };
}
