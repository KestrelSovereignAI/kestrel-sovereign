/** Plain identity text shared by every native-action approval surface. */
export function nativeActionLabel(action) {
    return `${action.action} · ${action.method} · ${action.task_handle}`;
}
