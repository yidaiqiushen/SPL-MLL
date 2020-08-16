import xlwt


def save_result(dataset_name, metrics_test):
    style = xlwt.easyxf('font: name Times New Roman, color-index black, bold on', num_format_str='#,##0.00')
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Result_Test')

    ws.write(0, 0, 'Ranking Loss')
    ws.write(0, 1, 'Hamming Loss')
    ws.write(0, 2, 'Average Precision')
    ws.write(0, 3, 'Micro-F1')
    ws.write(0, 4, 'Macro-F1')

    ws.write(1, 0, metrics_test['ranking_loss'])
    ws.write(1, 1, metrics_test['hamming_loss'])
    ws.write(1, 2, metrics_test['average_precision'])
    ws.write(1, 3, metrics_test['micro_f1'])
    ws.write(1, 4, metrics_test['macro_f1'])

    wb.save(dataset_name + '.xls')

