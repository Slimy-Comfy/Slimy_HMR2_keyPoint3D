import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "Slimy.HMR2KeyPoint3D",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_HMR2KeyPoint3D") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const preview = ComfyWidgets["STRING"](
                this, "hmr2_preview",
                ["STRING", { multiline: true }],
                app
            ).widget;
            preview.inputEl.readOnly         = true;
            preview.inputEl.style.fontFamily  = "monospace";
            preview.inputEl.style.fontSize    = "11px";
            preview.value         = "Queue Promptを実行してください。";
            preview.inputEl.value = "Queue Promptを実行してください。";

            this.addWidget("button", "hmr2_save_btn", "💾 JSONを保存", () => {
                const text = this.widgets?.find(w => w.name === "hmr2_preview")?.value ?? "";
                if (!text || text.startsWith("Queue")) {
                    alert("まだデータがありません。先にQueue Promptを実行してください。");
                    return;
                }
                const d  = new Date();
                const ts = `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,"0")}${String(d.getDate()).padStart(2,"0")}_${String(d.getHours()).padStart(2,"0")}${String(d.getMinutes()).padStart(2,"0")}${String(d.getSeconds()).padStart(2,"0")}`;
                const blob = new Blob([text], { type: "application/json" });
                const url  = URL.createObjectURL(blob);
                const a    = document.createElement("a");
                a.href     = url;
                a.download = `hmr2_keypoint3d_${ts}.json`;
                a.click();
                URL.revokeObjectURL(url);
            });

            this.size = [this.size[0], 260];

            const onResize = this.onResize;
            this.onResize = function (size) {
                onResize?.apply(this, arguments);
                fitPreview(this);
            };

            fitPreview(this);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0];
            if (!text) return;
            const w = this.widgets?.find(w => w.name === "hmr2_preview");
            if (w) {
                w.value         = text;
                w.inputEl.value = text;
                fitPreview(this);
                app.graph.setDirtyCanvas(true, false);
            }
        };
    },
});

function fitPreview(node) {
    const w = node.widgets?.find(w => w.name === "hmr2_preview");
    if (!w?.inputEl) return;

    const nodeH       = node.size[1];
    const widgetCount = (node.widgets?.length ?? 0) - 1;
    const otherH      = widgetCount * 38 + 60;
    const available   = Math.max(100, nodeH - otherH);

    w.inputEl.style.height = `${available}px`;
}
