import helper
import evaluation_setup_creator
import create_test_setup
import train
import compute_metrics_calculation


def run_all():
    d_setup, __ = helper.load_setup()
    if (len(d_setup)) > 0:
        print "Creating Evaluation Setup: {0}".format(d_setup['evaluationSetupPath'])
        evaluation_setup_creator.main()
        print "Training Classifier: {0}".format(d_setup['classifierPath'])
        train.main()
        print "Creating Test Setup: {0}".format(d_setup['testSetupPath'])
        create_test_setup.main()
        print "Computing Metrics for: {0}".format(d_setup['classifierPath'])
        compute_metrics_calculation.main()
    print 'Task Completed'


if __name__ == '__main__':
    run_all()
