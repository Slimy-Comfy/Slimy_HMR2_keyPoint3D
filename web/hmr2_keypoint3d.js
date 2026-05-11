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

Click [💾 Download] after running to save to your PC.`;

            const preview = ComfyWidgets["STRING"](
                this,
                "hmr2_preview",
                ["STRING", { multiline: true }],
                app
            ).widget;

            preview.inputEl.readOnly = true;
            preview.inputEl.style.fontFamily = "monospace";
            preview.inputEl.style.fontSize = "11px";
            preview.value = helpText;
            preview.inputEl.value = helpText;

            this._hmr2_thumb_b64 = null;
            this._hmr2_output_format = "JSON";

            const updateBtnLabel = () => {

                const fmtWidget = this.widgets?.find(
                    w => w.name === "output_format"
                );

                const dlBtn = this.widgets?.find(
                    w =>
                        w.type === "button" &&
                        (
                            w.name?.includes("Download") ||
                            w.value?.includes("Download")
                        )
                );

                if (!fmtWidget || !dlBtn) return;

                const value = String(fmtWidget.value ?? "");

                const isJson =
                    value === "JSON" ||
                    value.includes("JSON only");

                const label = isJson
                    ? "💾 Download JSON"
                    : "💾 Download PNG";

                dlBtn.name = label;
                dlBtn.value = label;

                app.graph.setDirtyCanvas(true, false);
            };

            const downloadWidget = this.addWidget(
                "button",
                "💾 Download JSON",
                "💾 Download JSON",
                () => {

                    const text =
                        this.widgets?.find(
                            w => w.name === "hmr2_preview"
                        )?.value ?? "";

                    if (!text || text.startsWith("About")) {
                        alert("まだデータがありません。先にQueue Promptを実行してください。");
                        return;
                    }

                    const d = new Date();

                    const ts =
                        `${d.getFullYear()}`
                        + `${String(d.getMonth() + 1).padStart(2, "0")}`
                        + `${String(d.getDate()).padStart(2, "0")}_`
                        + `${String(d.getHours()).padStart(2, "0")}`
                        + `${String(d.getMinutes()).padStart(2, "0")}`
                        + `${String(d.getSeconds()).padStart(2, "0")}`;

                    const stem = `hmr2_keypoint3d_${ts}`;

                    const fmtWidget =
                        this.widgets?.find(
                            w => w.name === "output_format"
                        );

                    const value = String(fmtWidget?.value ?? "");

                    const isJson =
                        value === "JSON" ||
                        value.includes("JSON only");

                    if (isJson) {

                        const jsonBlob = new Blob(
                            [text],
                            { type: "application/json" }
                        );

                        const jsonUrl =
                            URL.createObjectURL(jsonBlob);

                        const a = document.createElement("a");

                        a.href = jsonUrl;
                        a.download = `${stem}.json`;

                        a.click();

                        URL.revokeObjectURL(jsonUrl);

                        return;
                    }

                    if (this._hmr2_thumb_b64) {

                        const pngUrl =
                            `data:image/png;base64,${this._hmr2_thumb_b64}`;

                        const a = document.createElement("a");

                        a.href = pngUrl;
                        a.download = `${stem}.png`;

                        a.click();

                    } else {

                        alert("サムネイルがありません。再実行してください。");
                    }
                }
            );

            downloadWidget.serialize = false;

            this.size = [Math.max(this.size[0], 320), 390];

            const hookFormatWidget = () => {

                const fmtWidget =
                    this.widgets?.find(
                        w => w.name === "output_format"
                    );

                if (!fmtWidget) return;

                if (fmtWidget._hmr2_hooked) {
                    updateBtnLabel();
                    return;
                }

                fmtWidget._hmr2_hooked = true;

                const origCallback = fmtWidget.callback;

                fmtWidget.callback = (...args) => {

                    origCallback?.apply(fmtWidget, args);

                    requestAnimationFrame(() => {
                        updateBtnLabel();
                    });
                };

                updateBtnLabel();
            };

            // ★ 修正：UI 初期化タイミングを 2 フレーム待つ
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    hookFormatWidget();
                });
            });

            const origOnAdded = this.onAdded;

            this.onAdded = function () {

                origOnAdded?.apply(this, arguments);

                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        hookFormatWidget();
                    });
                });
            };

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

            const text = message?.text?.[0] ?? message?.text;

            if (text) {

                const w = this.widgets?.find(
                    w => w.name === "hmr2_preview"
                );

                if (w) {

                    w.value = text;
                    w.inputEl.value = text;

                    fitPreview(this);

                    app.graph.setDirtyCanvas(true, false);
                }
            }

            const thumb = message?.thumbnail_b64?.[0] ?? message?.thumbnail_b64;

            if (thumb) {
                this._hmr2_thumb_b64 = thumb;
            }

            const fmt = message?.output_format?.[0] ?? message?.output_format;

            if (fmt) {
                this._hmr2_output_format = fmt;
            }
        };
    },
});

function fitPreview(node) {

    const w = node.widgets?.find(
        w => w.name === "hmr2_preview"
    );

    if (!w?.inputEl) return;

    const nodeH = node.size[1];

    const widgetCount =
        Math.max(0, (node.widgets?.length ?? 0) - 1);

    const otherH =
        widgetCount * 38 + 60;

    const available =
        Math.max(100, nodeH - otherH);

    w.inputEl.style.height = `${available}px`;
}
