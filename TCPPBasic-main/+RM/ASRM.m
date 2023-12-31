classdef ASRM < handle
    % ASRM Summary
    %   ASRM (ArmStoneReachabilityMap). Basic idea is that we have a lot of
    %   data stored in files  +RM/data/... see xxx_fn descriptions. 
    %   This data might be redundant. Also I used both sparse and dense
    %   arrays to store this data. I believe sparse array functionality is
    %   no longer used in this class as dense array proved to be faster.
    
    %   !!!
    %   Anyway, the basic problem this class is trying to solve is that we
    %   want a function that takes some point (xyz) in robot frame,
    %   also some desired end-effector orientation expressed as a unit
    %   vector and gives use the ReachabilityIndex (RI)...
    %    However, the problem is that we need to 1) take this xyz point and
    %    project it into the discrete grid that RM data is generated on,
    %    then 2) check which test poses correspond to the desired unit
    %    vector 3) sum up only those IK_exists_bools that correspond to
    %    those poses. All this is done in point2ri function.

    properties %(Access = private)
        BaseToArmOffset = [0.175, 0, -0.053 + 0.086 + 0.1005] %Basefootprint to xarm_mount
        %Files
        grid_fn = "+RM/data/xarm_grid_checked.mat" % a grid of  points in a sphere volume around the robot
        poses_fn = "+RM/data/poses_on_sphere_200.mat" % These are poses on a test sphere w.r.t sphere center
        ik_fn = "+RM/data/xarm_ik_200_bools.mat"  % These are bools that say if IK for all points on grid_fn and all poses in poses_fn exists
        sparse_rm_fn = "+RM/data/as_sparse_rm_200.mat" % same but sparse array I think
        sparse_rm_smooth_fn = "+RM/data/as_sparse_rm_200_to_7_kernel_smooth.mat"
        %Privates
        poses % test poses 
        zvecs
        xs % grid indexes
        ys % grid indexes
        zs % grid indexes
        n % num elem of of squeezed data
        nn %num elem of sparse arrays
        m %num elem of poses
        sparse_index % sparse nnx1 logical array. For easier non-zero element find
        sparse_bools % sparse nnxm logical array.  representing rm
        sparse_grid % sparse nnx3 double array. for storing points
        d % the vox difference
        angle_toll=15; % Default angle of cone used to check RI
    end

    methods

        function self = ASRM()
            % General
            grid_full = load(self.grid_fn).grid;
            grid_full = grid_full + self.BaseToArmOffset;
            self.poses = load(self.poses_fn).poses;
            iks = load(self.ik_fn).rm_bools;

            % indexing
            assert(all(diff(unique(grid_full(:, 1))) > 0.025))
            assert(all(diff(unique(grid_full(:, 2))) > 0.025))
            assert(all(diff(unique(grid_full(:, 3))) > 0.025))
            self.xs = unique(grid_full(:, 1));
            self.ys = unique(grid_full(:, 2));
            self.zs = unique(grid_full(:, 3));
            self.d = max([diff(self.xs); diff(self.ys); diff(self.zs)]);
            assert(all(diff(self.xs) > 0))
            assert(all(diff(self.ys) > 0))
            assert(all(diff(self.zs) > 0))

            % map
            self.nn = self.indexfun(length(self.xs), length(self.ys), length(self.zs));
            self.m = size(iks, 2);
            self.n = size(grid_full, 1);

            self.sparse_index = false(self.nn, 1);
            self.sparse_bools = false(self.nn, self.m);
            self.sparse_grid = zeros(self.nn, 3);

            for ii = 1:self.n
                idx = find(abs(self.xs - grid_full(ii, 1)) < 0.025);
                idy = find(abs(self.ys - grid_full(ii, 2)) < 0.025);
                idz = find(abs(self.zs - grid_full(ii, 3)) < 0.025);
                assert(all(size(idx) == 1));
                assert(all(size(idy) == 1));
                assert(all(size(idz) == 1));

                id = self.indexfun(idx, idy, idz);

                assert(~self.sparse_index(id)) % make sure no collision

                self.sparse_index(id) = true;
                self.sparse_bools(id, :) = iks(ii, :);
                self.sparse_grid(id, :) = grid_full(ii, :);

                display(round(ii / self.n, 3))
            end

            self.sparse_index = sparse(self.sparse_index);
            self.sparse_bools = sparse(self.sparse_bools);
            self.sparse_grid = sparse(self.sparse_grid);

            % prep for ri comps:
            temp = TForm.vec2tform(self.poses);
            self.zvecs = squeeze(temp(1:3, 3, :))';

            %

        end

        function erode(self)
            kernel = [1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1; 1 1 0; -1 -1 0; 1 -1 0; -1 1 0];
            kn = size(kernel, 1);

            temp_index = false(self.nn, 1);
            temp_bools = false(self.nn, self.m);
            temp_grid = zeros(self.nn, 3);

            [X, Y, Z] = ndgrid(1:length(self.xs), 1:length(self.ys), 1:length(self.zs));
            ix = X(:);
            iy = Y(:);
            iz = Z(:);

            for ii = 1:size(ix, 1)
                index = self.indexfun(ix(ii), iy(ii), iz(ii));
                k_is = false(1, kn);

                for k = 1:kn
                    kixyz = [ix(ii) iy(ii) iz(ii)] + kernel(k, :);
                    kindex = self.indexfun(kixyz(1), kixyz(2), kixyz(3));

                    if any(kixyz == 0) || any(kixyz > [length(self.xs) length(self.ys) length(self.zs)])
                        k_is(k) = false;
                    else
                        k_is(k) = self.sparse_index(kindex);
                    end

                end

                if sum(k_is) > kn - 3
                    temp_bools(index, :) = self.sparse_bools(index, :);

                    if ~any(temp_bools(index, :))
                        kk = find(k_is);
                        kk = kk(1);
                        kixyz = [ix(ii) iy(ii) iz(ii)] + kernel(kk, :);
                        kindex = self.indexfun(kixyz(1), kixyz(2), kixyz(3));
                        temp_bools(index, :) = self.sparse_bools(kindex, :);
                        temp_grid(index, :) = [self.xs(ix(ii)) self.ys(iy(ii)) self.zs(iz(ii))];
                        temp_index(index) = true;
                    else
                        temp_grid(index, :) = self.index2point(index);
                        temp_index(index) = true;
                    end

                end

                display(round(ii / size(ix, 1), 3))
            end

            self.sparse_index = sparse(temp_index);
            self.sparse_bools = sparse(temp_bools);
            self.sparse_grid = sparse(temp_grid);
            self.n = sum(self.sparse_index);

        end

        function smooth(self)
            % 11x smooth
            kernel = [0 0 0; 1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1; 1 1 0; -1 -1 0; 1 -1 0; -1 1 0];
            kn = size(kernel, 1);
            mold = self.m;
            mnew = kn * self.m;

            temp_index = false(self.nn, 1);
            temp_bools = false(self.nn, mnew);
            temp_grid = zeros(self.nn, 3);

            [X, Y, Z] = ndgrid(1:length(self.xs), 1:length(self.ys), 1:length(self.zs));
            ix = X(:);
            iy = Y(:);
            iz = Z(:);

            for ii = 1:size(ix, 1)
                index = self.indexfun(ix(ii), iy(ii), iz(ii));

                for k = 1:kn
                    kixyz = [ix(ii) iy(ii) iz(ii)] + kernel(k, :);
                    kindex = self.indexfun(kixyz(1), kixyz(2), kixyz(3));

                    if any(kixyz == 0) || any(kixyz > [length(self.xs) length(self.ys) length(self.zs)])
                        temp_bools(index, 1 + (k - 1) * mold:k * mold) = false(1, mold);
                    else
                        temp_bools(index, 1 + (k - 1) * mold:k * mold) = self.sparse_bools(kindex, :);
                    end

                end

                if any(temp_bools(index, :))
                    temp_grid(index, :) = [self.xs(ix(ii)) self.ys(iy(ii)) self.zs(iz(ii))];
                    temp_index(index) = true;
                end

                display(round(ii / size(ix, 1), 3))
            end

            self.sparse_index = sparse(temp_index);
            self.sparse_bools = sparse(temp_bools);
            self.sparse_grid = sparse(temp_grid);

            self.m = mnew;
            self.n = sum(self.sparse_index);
            self.poses = repmat(self.poses, 11, 1);
            self.zvecs = repmat(self.zvecs, 11, 1);

        end
        
%         go from grid index into array index (essentially hashmap)
        function index = indexfun(self, xi, yi, zi)
            index = xi * 10000 + yi * 100 + zi;
        end

        %% Point2_
        function bools = point2bools(self, points)
            bools = false(size(points, 1), self.m);
            index = self.point2index(points);

            for ii = 1:size(points, 1)

                if index(ii) < 0
                    bools(ii, :) = false(1, self.m);
                else
                    bools(ii, :) = self.sparse_bools(index(ii), :);
                end

            end

        end

        function ris = point2ri(self, points, varargin)
            % vararg in same as comp ri
            index = self.point2index(points);
            ris = full(self.comp_ri(self.sparse_bools(index, :), varargin{:}));

        end

        function index = point2index(self, points)
            %This should be in base footprint frame
            index = zeros(size(points, 1), 1);

            for ii = 1:size(index, 1)
                idx = find(abs(self.xs - points(ii, 1)) <= self.d);
                idy = find(abs(self.ys - points(ii, 2)) <= self.d);
                idz = find(abs(self.zs - points(ii, 3)) <= self.d);

                if ~isempty(idx) && ~isempty(idy) && ~isempty(idz)
                    [c, i] = min(abs(self.xs(idx) - points(ii, 1)));
                    idx = idx(i);
                    [c, i] = min(abs(self.ys(idy) - points(ii, 2)));
                    idy = idy(i);
                    [c, i] = min(abs(self.zs(idz) - points(ii, 3)));
                    idz = idz(i);
                    index(ii) = self.indexfun(idx, idy, idz);
                else
                    index(ii) = 1;
                end

            end

        end

        %% Index2_
        function point = index2point(self, index)
            point = self.sparse_grid(index, :);
            assert(any(point))
        end

        function bools = index2rbools(self, index)
            bools = self.sparse_bools(index, :);
        end

        function ri = index2ri(self, index, varargin)
            ri = self.comp_ri(self.sparse_bools(index, :), varargin{:});
        end

        %% ri
        function ri = comp_ri(self, bools, varargin)
            %bools is just 1xm vec. varargin 1 is 1x3 vec. vararg 2 is
            %tolerance in degree.
            nInputs = numel(varargin);
            angle_tol = self.angle_toll;

            if nInputs == 0
                ri = sum(bools, 2) ./ self.m;
            elseif all(size(varargin{1}) == [1, 3]) && (nInputs == 1 || nInputs == 2)

                if nInputs == 2
                    angle_tol = varargin{2};
                end

                dir_ = varargin{1};

                if size(dir_, 2) > size(dir_, 1)
                    dir_ = dir_';
                end

                dotproduct = self.zvecs * dir_;
                angle_bools = acosd(dotproduct) < angle_tol;
                reduced_bools = (bools & (angle_bools')); %no need to divide cuz both norms are one
                ri = sum(reduced_bools, 2) / sum(angle_bools);
            else
                display("Function inputs not supported. comp_ri")
            end

        end

        function [points, ris] = get_rm(self, varargin)
            %varargin 1 is 1x3 vec. vararg 2 is
            %tolerance in degree.
            ris = self.comp_ri(self.sparse_bools(self.sparse_index, :), varargin{:});
            points = self.sparse_grid(self.sparse_index, :);

        end

        %% TESTS
        function test_pt2id2pt(self)

            for ii = 1:size(self.grid, 1)
                point = self.grid(ii, :);
                ids = self.point2index(point);
                retrieved_pt = self.index2point(ids);
                assert(all(point == retrieved_pt));
                round(ii / size(self.grid, 1), 2)
            end

        end

        function test_id2pt2id(self)

            for id = 1:size(self.rm, 1)
                point = self.index2point(id);

                if isempty(point)
                    continue
                end

                id_retrieved = self.point2index(point);
                assert(id == id_retrieved);
                round(id / size(self.rm, 1), 2)
            end

        end

    end

end
