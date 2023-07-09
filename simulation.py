from libraries import *
from perf_measures import *

# GAN Training
#
# Inputs:
#    epoch: epoch #
#    generator: generator network
#    generator_optimizer: generator optimizer
#    discriminator: discriminator network
#    discriminator_optimizer: discriminator optimizer
#
# Outputs:
#    average_generator_loss: binary cross entropy (scalar)
#    average_discriminator_loss: binary cross entropy (scalar)
#
def gan_train(epoch, generator, generator_optimizer, discriminator, discriminator_optimizer, generator_loss_function, discriminator_loss_function, train_loader, device):
    discriminator_loss = 0
    generator_loss = 0
    calender_count = 0
    butterfly_count = 0
    discriminator_detected_calendar_arbitrage = 0
    discriminator_detected_butterfly_arbitrage = 0
    for batch_idx, (X, y) in enumerate(train_loader):
        X.requires_grad = True
        X = X.to(device).float()
        y = y.to(device).float()
        # train discriminator with all real
        discriminator_optimizer.zero_grad()
        y_ = torch.column_stack((X, y.unsqueeze(1)))
        pred_real = discriminator(y_)
        label_real = torch.full(pred_real.size(), 1, dtype=torch.float, device=device)
        dloss_real = discriminator_loss_function(pred_real, label_real)
        dloss_real.backward()
        # train discriminator with noise
        z_ = torch.column_stack((torch.rand(X.size()), generator(torch.rand(X.size()))))
        # z_ = torch.column_stack((X, generator(X).detach()))
        pred_fake = discriminator(z_)
        label_fake = torch.full(pred_fake.size(), 0, dtype=torch.float, device=device)
        dloss_fake = discriminator_loss_function(pred_fake, label_fake)
        dloss_fake.backward()
        discriminator_optimizer.step()
        discriminator_loss += (dloss_real + dloss_fake).item() # dloss_gen.item() #
        # update generator
        generator_optimizer.zero_grad()
        z_ = torch.column_stack((X, generator(X)))
        pred_gen = discriminator(z_)
        label_gen = torch.full(pred_gen.size(), 1, dtype=torch.float, device=device)
        gloss_gen = discriminator_loss_function(pred_gen, label_gen)
        gloss, c4_loss, c5_loss = generator_loss_function(X, generator(X), y.unsqueeze(1), gloss_gen)
        gloss.backward()
        torch.nn.utils.clip_grad_value_(generator.parameters(), 1)
        generator_optimizer.step()
        generator_loss += gloss.item()
        # add the number of arbitrage violations
        calender_count += torch.count_nonzero(c4_loss).item()
        butterfly_count += torch.count_nonzero(c5_loss).item()
        
        # get number of arbitrage violation detected by the discriminator
        discriminator_class = torch.where(pred_gen > 0.5, 1, 0).squeeze(1)
        calendar_arbitrage_cases = torch.where(c4_loss > 0, 1, 0)
        butterfly_arbitrage_cases = torch.where(c5_loss > 0, 1, 0)
        discriminator_detected_calendar_arbitrage += torch.count_nonzero(torch.where((discriminator_class+calendar_arbitrage_cases) == 2, 1, 0)).item()
        discriminator_detected_butterfly_arbitrage += torch.count_nonzero(torch.where((discriminator_class+butterfly_arbitrage_cases) == 2, 1, 0)).item()

    average_discriminator_loss = discriminator_loss / len(train_loader.dataset)
    average_generator_loss = generator_loss / len(train_loader.dataset)
    return (average_generator_loss, average_discriminator_loss, calender_count, butterfly_count, 
            discriminator_detected_calendar_arbitrage, discriminator_detected_butterfly_arbitrage)

# GAN Test
#
# Have a look at the following tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
#
# Inputs:
#    epoch: epoch #
#    generator: generator network
#    discriminator: discriminator network
#
# Outputs:
#    average_generator_loss: binary cross entropy (scalar)
#    average_discriminator_loss: binary cross entropy (scalar)
#    generator_mae, generator_mape: performance measure
#
def gan_test(epoch, generator, discriminator, generator_loss_function, discriminator_loss_function, test_loader, device):
    discriminator_loss = 0
    generator_loss = 0
    calender_count = 0
    butterfly_count = 0
    discriminator_detected_calendar_arbitrage = 0
    discriminator_detected_butterfly_arbitrage = 0

    generator_mae = 0
    generator_mape = 0
    discriminator_mae = 0

    discriminator.eval()
    generator.eval()
    for batch_idx, (X, y) in enumerate(test_loader):
        X.requires_grad = True
        X = X.to(device).float()
        y = y.to(device).float()
        # discriminator with real data
        y_ = torch.column_stack((X, y.unsqueeze(1)))
        pred_real = discriminator(y_)
        label_real = torch.full(pred_real.size(), 1, dtype=torch.float, device=device)
        dloss_real = discriminator_loss_function(pred_real, label_real)
        # discriminator with noise
        z_ = torch.column_stack((X, generator(X).detach()))
        pred_fake = discriminator(z_)
        label_fake = torch.full(pred_fake.size(), 0, dtype=torch.float, device=device)
        dloss_fake = discriminator_loss_function(pred_fake, label_fake)
        discriminator_loss += (dloss_real + dloss_fake).item()
        # generator
        z_ = torch.column_stack((X, generator(X)))
        pred_gen = discriminator(z_)
        label_gen = torch.full(pred_gen.size(), 1, dtype=torch.float, device=device)
        gloss_gen = discriminator_loss_function(pred_gen, label_gen)
        gloss, c4_loss, c5_loss = generator_loss_function(X, generator(X), y.unsqueeze(1), gloss_gen)
        generator_loss += gloss.item()
        
        # add the number of arbitrage violations
        calender_count += torch.count_nonzero(c4_loss).item()
        butterfly_count += torch.count_nonzero(c5_loss).item()

        # get number of arbitrage violation detected by the discriminator
        discriminator_class = torch.where(pred_gen > 0.5, 1, 0).squeeze(1)
        calendar_arbitrage_cases = torch.where(c4_loss > 0, 1, 0)
        butterfly_arbitrage_cases = torch.where(c5_loss > 0, 1, 0)
        discriminator_detected_calendar_arbitrage += torch.count_nonzero(torch.where((discriminator_class+calendar_arbitrage_cases) == 2, 1, 0)).item()
        discriminator_detected_butterfly_arbitrage += torch.count_nonzero(torch.where((discriminator_class+butterfly_arbitrage_cases) == 2, 1, 0)).item()

        # add perfomance measure
        discriminator_mae += mean_abs_error(pred_real.detach()+pred_gen.detach(), label_real+label_gen) # mean_abs_error(pred_gen.detach(), pred_real.detach()) # 
        generator_mae += mean_abs_error(generator(X).detach(), y.unsqueeze(1))
        generator_mape += mean_abs_percentage_error(generator(X).detach(), y.unsqueeze(1))

    average_discriminator_loss = discriminator_loss / len(test_loader.dataset)
    average_generator_loss = generator_loss / len(test_loader.dataset)
    discriminator_mae = discriminator_mae / len(test_loader.dataset)
    generator_mae = generator_mae / len(test_loader.dataset)
    generator_mape = generator_mape / len(test_loader.dataset)
    return (average_generator_loss, average_discriminator_loss, calender_count, butterfly_count, 
            discriminator_detected_calendar_arbitrage, discriminator_detected_butterfly_arbitrage, 
            discriminator_mae, generator_mae, generator_mape)