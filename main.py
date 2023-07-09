from libraries import *
from gan_functions import *
from simulation import *
from loss_functions import *
from data_split import *

discriminator_loss_function = BCELoss()
generator_loss_function = RMSENoArbitrageLoss(penalty = [1, 10, 10, 10, 1])

# train and test gan
epochs = 50
discriminator_average_train_losses = []
discriminator_average_test_losses = []
generator_average_train_losses = []
generator_average_test_losses = []

# arbitrage metrics
calender_train_counts = []
calender_test_counts = []
butterfly_train_counts = []
butterfly_test_counts = []
discriminator_detected_calendar_arbitrages_train = []
discriminator_detected_butterfly_arbitrages_train = []
discriminator_detected_calendar_arbitrages_test = []
discriminator_detected_butterfly_arbitrages_test = []

# eval performance metrics
generator_test_maes_constraints = []
generator_test_mapes_constraints = []
discriminator_test_maes = []

generator_model_constraint = Generator().to(device)
generator_optimizer = optim.Adam(generator_model_constraint.parameters(), lr=1e-3)
discriminator_model_constraint = Discriminator().to(device)
discriminator_optimizer = optim.Adam(discriminator_model_constraint.parameters(), lr=1e-3)

start_time = datetime.datetime.now()
for epoch in range(1, epochs + 1):
    (generator_average_train_loss, 
     discriminator_average_train_loss, 
     calender_train_count, 
     butterfly_train_count, 
     discriminator_detected_calendar_arbitrage_train,
     discriminator_detected_butterfly_arbitrage_train) = gan_train(epoch, 
                                                                   generator_model_constraint, 
                                                                   generator_optimizer, 
                                                                   discriminator_model_constraint, 
                                                                   discriminator_optimizer,generator_loss_function,discriminator_loss_function,
                                                                   train_loader,
                                                                   device)
    discriminator_average_train_losses.append(discriminator_average_train_loss)
    generator_average_train_losses.append(generator_average_train_loss)
    
    (generator_average_test_loss, 
     discriminator_average_test_loss, 
     calender_test_count, 
     butterfly_test_count, 
     discriminator_detected_calendar_arbitrage_test, 
     discriminator_detected_butterfly_arbitrage_test, 
     discriminator_mae, 
     generator_mae, 
     generator_mape) = gan_test(epoch, 
                                generator_model_constraint, 
                                discriminator_model_constraint,
                                generator_loss_function,
                                discriminator_loss_function,
                                test_loader,
                                device)
    discriminator_average_test_losses.append(discriminator_average_test_loss)
    generator_average_test_losses.append(generator_average_test_loss)
    
    # add arbitrage metrics
    calender_train_counts.append(calender_train_count)
    calender_test_counts.append(calender_test_count)
    butterfly_train_counts.append(butterfly_train_count)
    butterfly_test_counts.append(butterfly_test_count)
    discriminator_detected_calendar_arbitrages_train.append(discriminator_detected_calendar_arbitrage_train)
    discriminator_detected_butterfly_arbitrages_train.append(discriminator_detected_butterfly_arbitrage_train)
    discriminator_detected_calendar_arbitrages_test.append(discriminator_detected_calendar_arbitrage_test)
    discriminator_detected_butterfly_arbitrages_test.append(discriminator_detected_butterfly_arbitrage_test)

    # add the eval metrics
    generator_test_maes_constraints.append(generator_mae)
    generator_test_mapes_constraints.append(generator_mape)
    discriminator_test_maes.append(discriminator_mae)