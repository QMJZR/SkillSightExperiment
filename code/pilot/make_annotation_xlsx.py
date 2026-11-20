import argparse
import csv

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


WIDTHS = {
    "pair_id": 9,
    "q_block": 9,
    "query": 60,
    "reference_answer": 28,
    "skill_name": 28,
    "skill_description": 50,
    "skill_snippet": 90,
    "semantic": 10,
    "coverage": 10,
    "judgment": 12,
    "invalid_reason": 15,
    "evidence": 40,
    "confidence": 10,
    "notes": 30,
    "annotator": 10,
}
WRAP = {"query", "reference_answer", "skill_description", "skill_snippet", "evidence", "notes"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--xlsx", required=True)
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())

    wb = Workbook()
    ws = wb.active
    ws.title = "标注"
    ws.append(fields)
    header_fill = PatternFill("solid", fgColor="1A3A7A")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in rows:
        ws.append([row.get(f, "") for f in fields])

    for idx, field in enumerate(fields, start=1):
        letter = get_column_letter(idx)
        ws.column_dimensions[letter].width = WIDTHS.get(field, 14)
        for cell in ws[letter][1:]:
            cell.alignment = Alignment(
                vertical="top", wrap_text=field in WRAP, horizontal="left" if field in WRAP else "center"
            )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    def add_validation(field, options, prompt):
        col = get_column_letter(fields.index(field) + 1)
        dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True)
        dv.error = "请从下拉列表中选择"
        dv.errorTitle = "取值不合法"
        dv.prompt = prompt
        dv.promptTitle = field
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{len(rows) + 1}")

    add_validation("semantic", ["0", "1", "2"], "0 无关；1 弱相关；2 强相关")
    add_validation("coverage", ["0", "1", "2"], "0 不能完成；1 部分完成；2 全部完成")
    add_validation("judgment", ["Gold", "FVA", "Invalid"], "Gold / 功能等价替代 / 无效")
    add_validation("invalid_reason", ["generic", "topic_related", "unrelated", "duplicate"], "只在 Invalid 时填")
    add_validation("confidence", ["1", "2", "3"], "1 不确定；2 一般；3 确定")

    info = wb.create_sheet("说明")
    lines = [
        ["字段", "取值", "说明"],
        ["semantic", "0 / 1 / 2", "0 无关；1 弱相关，同领域但不同问题；2 强相关，主题和对象都对得上"],
        ["coverage", "0 / 1 / 2", "0 不能完成；1 只能完成部分必要步骤；2 能完成全部必要步骤"],
        ["judgment", "Gold / FVA / Invalid", "Gold 是数据集标注；FVA 是功能等价的替代；Invalid 是其余"],
        ["invalid_reason", "generic / topic_related / unrelated / duplicate", "只在 Invalid 时填"],
        ["evidence", "原文一句", "支持 coverage 判断的技能原文，必须能定位"],
        ["confidence", "1 / 2 / 3", "1 不确定；2 一般；3 确定"],
        ["", "", ""],
        ["FM 定义", "Invalid + semantic=2 + coverage=0", "主题强相关，但缺关键步骤或方法不对"],
        ["Partial FM", "Invalid + semantic=2 + coverage=1", "只能完成部分必要步骤，单独统计"],
        ["FVA", "非 Gold + coverage=2 + 不需要额外技能", "功能等价替代，不算失败"],
    ]
    for line in lines:
        info.append(line)
    for cell in info[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
    info.column_dimensions["A"].width = 18
    info.column_dimensions["B"].width = 42
    info.column_dimensions["C"].width = 90
    for row in info.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    wb.save(args.xlsx)
    print(f"wrote {args.xlsx} rows={len(rows)}")


if __name__ == "__main__":
    main()
