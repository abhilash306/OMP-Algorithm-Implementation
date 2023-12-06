function sparse_signal_recovery()
    % Prompt user for input parameters
    signalLength = input('Please enter value of signal length (n): ');
    probability = input('Please enter value of probability (p): ');
    errorThreshold = input('Please enter value of threshold (e): ');

    % Generate input signal
    inputSignal = generateBernoulliSequence(signalLength, probability);
    sparseSignal = generateSparseSignal(inputSignal);

    % Display Binomial Distribution for Sparsity
    displayBinomialDistribution(signalLength, probability);

    % Initialize recovery error and sparsity of recovered signals
    recoveryError = [];
    recoveredSparsity = [];

    % Perform recovery for different numbers of measurements
    for measurements = [200, 400, 600]
        [recoveredSignal, currentError] = performOMPRecovery(sparseSignal, measurements, errorThreshold);
        
        % Plot original and recovered signals
        plotSignals(inputSignal, recoveredSignal, signalLength, measurements, probability);

        % Update recovery error and sparsity
        recoveryError = [recoveryError, currentError];
        recoveredSparsity = [recoveredSparsity, nnz(recoveredSignal)];
    end

    % Display final results
    disp('Recovery Error:');
    disp(recoveryError);
    disp('Recovered Sparsity:');
    disp(recoveredSparsity);
    disp('Original Sparsity:');
    disp(sum(inputSignal));
end

function A = generateBernoulliSequence(n, p)
    A = (rand(n, 1) < p);
end

function X = generateSparseSignal(A)
    n = length(A);
    X = double.empty(n, 0);
    for i = 1:n
        if A(i) == 1
            X(i, 1) = randn;
        else
            X(i, 1) = 0;
        end
    end
end

function displayBinomialDistribution(n, p)
    h = 0:n;
    q = binopdf(h, n, p);
    figure;
    bar(h, q, 1);
    xlabel('Sparsity Level');
    ylabel('Probability');
    title(['N = ', num2str(n), ', p = ', num2str(p)]);
end

function [Xhat, error] = performOMPRecovery(X, m, e)
    n = length(X);
    phi = randn(m, n);
    y = phi * X;

    % OMP Recovery
    % Initialization
    k = 0;
    Xhat = zeros(n, 1);
    yhat = zeros(m, 1);
    rhat = y;
    lambda = [];

    % Iterative Recovery Loop
    while norm(rhat) > e
        k = k + 1;
        [value, location] = max(abs(phi' * rhat));
        lambda = [lambda, location];
        phi_k = phi(:, lambda);
        x_k = lsqminnorm(phi_k, y);
        for i = 1:length(lambda)
            Xhat(lambda(i)) = x_k(i);
        end
        yhat = phi * Xhat;
        rhat = y - yhat;
    end

    % Compute recovery error
    error = norm(X - Xhat);
end

function plotSignals(A, Xhat, n, m, p)
    figure; hold on; grid on;
    stem(1:length(A), A, 'r');
    stem(1:length(A), Xhat, 'b');
    legend('Original Signal', 'Recovered Signal');
    title(['N = ', num2str(n), ', M = ', num2str(m), ', p = ', num2str(p)]);
end
