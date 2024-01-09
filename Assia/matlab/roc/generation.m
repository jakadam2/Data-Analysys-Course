
function[points1, points0] = generation(m0, sigma, N)

center = [0, 0];  % Center of the cloudS

points1 = sqrt(sigma) * randn(N, 2);
points1 = points1 + center;

center = [m0, 0];

points0 = sqrt(sigma) * randn(N, 2);
points0 = points0 + center;


% scatter(points1(:, 1), points1(:, 2), 'o');
% 
% hold on 

% scatter(points0(:, 1), points0(:, 2), 'x');
% title('Plot classi');
% xlabel('X');
% ylabel('Y');
% axis equal; 