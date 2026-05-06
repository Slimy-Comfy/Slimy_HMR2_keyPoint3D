import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "Slimy.HMR2KeyPoint3D",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_HMR2KeyPoint3D") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const helpText = `About person_index:
  0        → All people detected (up to 10)
  1        → Person #1 only
  2,5,6    → Persons #2, #5 and #6 only
             (Which index is who requires trial and error)

Click [💾 Download JSON] after running to save
JSON + thumbnail (JPG) to your PC.`;

            const preview = ComfyWidgets["STRING"](
                this, "hmr2_preview",
                ["STRING", { multiline: true }],
                app
            ).widget;
            preview.inputEl.readOnly         = true;
            preview.inputEl.style.fontFamily  = "monospace";
            preview.inputEl.style.fontSize    = "11px";
            preview.value         = helpText;
            preview.inputEl.value = helpText;

            this._hmr2_thumb_b64 = null;

            this.addWidget("button", "💾 Download JSON", "💾 Download JSON", () => {
                const text = this.widgets?.find(w => w.name === "hmr2_preview")?.value ?? "";
                if (!text || text.startsWith("Queue")) {
                    alert("まだデータがありません。先にQueue Promptを実行してください。");
                    return;
                }
                const d  = new Date();
                const ts = `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,"0")}${String(d.getDate()).padStart(2,"0")}_${String(d.getHours()).padStart(2,"0")}${String(d.getMinutes()).padStart(2,"0")}${String(d.getSeconds()).padStart(2,"0")}`;
                const stem = `hmr2_keypoint3d_${ts}`;

                // JSON DL
                const jsonBlob = new Blob([text], { type: "application/json" });
                const jsonUrl  = URL.createObjectURL(jsonBlob);
                const jsonA    = document.createElement("a");
                jsonA.href     = jsonUrl;
                jsonA.download = `${stem}.json`;
                jsonA.click();
                URL.revokeObjectURL(jsonUrl);

                // PNG DL（サムネイルがあれば）
                if (this._hmr2_thumb_b64) {
                    const jpgUrl = `data:image/jpeg;base64,${this._hmr2_thumb_b64}`;
                    const jpgA   = document.createElement("a");
                    jpgA.href     = jpgUrl;
                    jpgA.download = `${stem}.jpg`;
                    jpgA.click();
                }
            });

            this.size = [this.size[0], 390];

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
            // サムネイルBase64を保持
            const thumb = message?.thumbnail_b64?.[0];
            if (thumb) {
                this._hmr2_thumb_b64 = thumb;
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
