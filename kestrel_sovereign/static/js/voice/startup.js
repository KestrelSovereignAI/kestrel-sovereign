/** Public startup cancellation is not a native acquisition completion receipt.
 * Keep the original operation rooted: permission/worklet loading may finish
 * late and MUST still run the client's original late-acquisition cleanup.
 */
export function createStartupSettlement() {
  let nativeStartup = null;
  let nativeSettlement = null;
  let closedError = null;
  let rejectClosed;
  const closed = new Promise((resolve, reject) => { rejectClosed = reject; });
  closed.catch(() => {});
  return {
    run(operation) {
      if (closedError) return Promise.reject(closedError);
      if (nativeStartup) return Promise.reject(new Error('Voice client already started.'));
      nativeStartup = Promise.resolve().then(operation);
      nativeSettlement = Promise.allSettled([nativeStartup]);
      // This settles the PUBLIC caller promptly. It neither cancels nor
      // declares completion of the original retained native operation.
      return Promise.race([nativeStartup, closed]);
    },
    whenSettled() { return nativeSettlement || Promise.resolve([]); },
    close(error = new Error('Original voice startup was closed.')) {
      if (closedError) return closedError;
      closedError = error;
      rejectClosed(closedError);
      return closedError;
    },
  };
}
