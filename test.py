from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter(opt)
    for i, data in enumerate(dataset):
        #print(i)
        model.set_input(data)
        ncorrect, nexamples, mean_iou, iou = model.test()
        writer.update_counter(ncorrect, nexamples, mean_iou, iou)
    writer.print_acc(epoch, writer.acc)
    writer.print_iou(epoch, writer.mean_iou, writer.seg_iou)
    return writer.acc, writer.mean_iou, writer.iou


if __name__ == '__main__':
    run_test()
