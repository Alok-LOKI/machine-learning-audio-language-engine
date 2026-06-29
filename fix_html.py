with open('static/notebook.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re
# Remove the old inject block
html = re.sub(r'<style id="custom-transparency-fix">.*?</style>', '', html, flags=re.DOTALL)
# Or if it doesn't have an ID, we'll just append after. The previous one might still be there, 
# but CSS cascading means the last one wins. We'll use !important.

inject = """
<style id="custom-transparency-fix">
:root, body, body[data-theme="light"], body[data-theme="dark"], .jp-Notebook {
    /* Make backgrounds transparent */
    --jp-layout-color0: transparent !important;
    --jp-layout-color1: transparent !important;
    --jp-layout-color2: rgba(255, 255, 255, 0.05) !important;
    --jp-layout-color3: rgba(255, 255, 255, 0.1) !important;
    --jp-cell-editor-background: transparent !important;
    --jp-cell-editor-active-background: rgba(255,255,255,0.02) !important;
    
    /* Make base text white/gray for readability on dark background */
    --jp-ui-font-color0: #ffffff !important;
    --jp-ui-font-color1: #eeeeee !important;
    --jp-ui-font-color2: #cccccc !important;
    --jp-content-font-color0: #ffffff !important;
    --jp-content-font-color1: #eeeeee !important;
    --jp-content-font-color2: #cccccc !important;
    
    background: transparent !important;
    background-color: transparent !important;
}

.jp-Cell, .jp-OutputArea, .jp-OutputArea-output, pre, .jp-InputArea-editor, .jp-CodeCell {
    background: transparent !important;
    background-color: transparent !important;
}

/* Force standard text to white so it's readable, but do NOT target `.cm-` or Pygments spans so syntax highlighting stays! */
.jp-RenderedHTMLCommon p, 
.jp-RenderedHTMLCommon li, 
.jp-RenderedHTMLCommon h1, 
.jp-RenderedHTMLCommon h2, 
.jp-RenderedHTMLCommon h3 {
    color: #ffffff !important;
}
.jp-OutputArea-output pre {
    color: #ffffff !important;
}
</style>
"""

html = html.replace('</head>', inject + '</head>')

with open('static/notebook.html', 'w', encoding='utf-8') as f:
    f.write(html)
