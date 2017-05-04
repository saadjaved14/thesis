import compute_metrics_calculation
import create_test_setup
import evaluation_setup_creator
import helper
import train


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
