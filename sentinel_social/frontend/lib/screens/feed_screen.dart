import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:ui';
import 'package:google_fonts/google_fonts.dart';
import 'upload_screen.dart';

const kBg = Color(0xFF09090B);
const kCard = Color(0xFF141416);
const kSurface = Color(0xFF1C1C1F);
const kBorder = Color(0xFF27272A);
const kAccent = Color(0xFFC9A84C);
const kAccent2 = Color(0xFFF0D78C);
const kMuted = Color(0xFF71717A);

const _posts = [
  {
    'user': 'alex.rivera',
    'avatar': 'https://i.pravatar.cc/150?img=11',
    'verified': true,
    'image': 'https://picsum.photos/seed/meme1/800/600',
    'likes': '12.4K',
    'caption': 'When the WiFi drops mid-Zoom call 😭 #relatable #wfh',
    'comments': 142,
    'time': '2 hours ago',
    'safe': true,
  },
  {
    'user': 'meme.central',
    'avatar': 'https://i.pravatar.cc/150?img=33',
    'verified': false,
    'image': 'https://picsum.photos/seed/meme2/800/600',
    'likes': '8.1K',
    'caption': 'Monday morning productivity vs Friday evening 😂 #mood',
    'comments': 89,
    'time': '5 hours ago',
    'safe': true,
  },
  {
    'user': 'sentinelx.ai',
    'avatar': 'https://i.pravatar.cc/150?img=57',
    'verified': true,
    'image': 'https://picsum.photos/seed/meme3/800/600',
    'likes': '31.2K',
    'caption': 'AI keeping the feed clean, one scan at a time 🛡️✨ #SentinelX',
    'comments': 301,
    'time': '1 day ago',
    'safe': true,
  },
];

const _stories = [
  {
    'name': 'Your Story',
    'avatar': 'https://i.pravatar.cc/150?img=5',
    'isYou': true,
  },
  {
    'name': 'alex.r',
    'avatar': 'https://i.pravatar.cc/150?img=11',
    'isYou': false,
  },
  {
    'name': 'meme.cx',
    'avatar': 'https://i.pravatar.cc/150?img=33',
    'isYou': false,
  },
  {
    'name': 'j.smith',
    'avatar': 'https://i.pravatar.cc/150?img=22',
    'isYou': false,
  },
  {
    'name': 'priya_k',
    'avatar': 'https://i.pravatar.cc/150?img=44',
    'isYou': false,
  },
  {
    'name': 'luca.d',
    'avatar': 'https://i.pravatar.cc/150?img=68',
    'isYou': false,
  },
];

class FeedScreen extends StatefulWidget {
  const FeedScreen({super.key});

  @override
  State<FeedScreen> createState() => _FeedScreenState();
}

class _FeedScreenState extends State<FeedScreen> {
  final Set<int> _liked = {};
  int _activeNav = 0;

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: SystemUiOverlayStyle.light,
      child: Scaffold(
        backgroundColor: kBg,
        body: CustomScrollView(
          slivers: [
            _buildAppBar(context),
            SliverToBoxAdapter(child: _buildStoriesRow()),
            SliverToBoxAdapter(
              child: Divider(color: kBorder.withOpacity(0.5), thickness: 0.3, height: 0.3),
            ),
            SliverList(
              delegate: SliverChildBuilderDelegate(
                (ctx, i) => _PostCard(
                  post: _posts[i],
                  isLiked: _liked.contains(i),
                  onLike:
                      () => setState(() {
                        _liked.contains(i) ? _liked.remove(i) : _liked.add(i);
                      }),
                ),
                childCount: _posts.length,
              ),
            ),
            const SliverToBoxAdapter(child: SizedBox(height: 80)),
          ],
        ),
        extendBody: true,
        bottomNavigationBar: _buildBottomNav(context),
      ),
    );
  }

  SliverAppBar _buildAppBar(BuildContext context) {
    return SliverAppBar(
      pinned: true,
      backgroundColor: kBg.withOpacity(0.85),
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      flexibleSpace: ClipRRect(
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(color: Colors.transparent),
        ),
      ),
      title: ShaderMask(
        shaderCallback:
            (bounds) => const LinearGradient(
              colors: [kAccent, kAccent2],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ).createShader(bounds),
        child: Text(
          'Sentinel',
          style: GoogleFonts.outfit(
            fontSize: 26,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.add_box_outlined, size: 25),
          onPressed:
              () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const UploadScreen()),
              ),
          color: Colors.white70,
          tooltip: 'New Post',
        ),
        Stack(
          children: [
            IconButton(
              icon: const Icon(Icons.favorite_border, size: 25),
              onPressed: () {},
              color: Colors.white70,
            ),
            Positioned(
              right: 9,
              top: 9,
              child: Container(
                width: 7,
                height: 7,
                decoration: BoxDecoration(
                  color: const Color(0xFFDC2626),
                  shape: BoxShape.circle,
                  border: Border.all(color: kBg, width: 1.5),
                ),
              ),
            ),
          ],
        ),
        IconButton(
          icon: const Icon(Icons.send_outlined, size: 25),
          onPressed: () {},
          color: Colors.white70,
        ),
        const SizedBox(width: 4),
      ],
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(0.3),
        child: Divider(color: kBorder.withOpacity(0.5), height: 0.3, thickness: 0.3),
      ),
    );
  }

  Widget _buildStoriesRow() {
    return SizedBox(
      height: 116,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        itemCount: _stories.length,
        itemBuilder: (ctx, i) {
          final s = _stories[i];
          final isYou = s['isYou'] as bool;
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 7),
            child: Column(
              children: [
                Stack(
                  alignment: Alignment.center,
                  children: [
                    Container(
                      width: 70,
                      height: 70,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient:
                            isYou
                                ? null
                                : const LinearGradient(
                                  colors: [
                                    Color(0xFFC9A84C),
                                    Color(0xFFE8C86A),
                                    Color(0xFFF0D78C),
                                  ],
                                  begin: Alignment.bottomLeft,
                                  end: Alignment.topRight,
                                ),
                        color: isYou ? kSurface : null,
                        border:
                            isYou
                                ? Border.all(color: kBorder, width: 1.5)
                                : null,
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(2.5),
                        child: Container(
                          decoration: const BoxDecoration(
                            shape: BoxShape.circle,
                            color: kBg,
                          ),
                          padding: const EdgeInsets.all(1.5),
                          child: CircleAvatar(
                            backgroundImage: NetworkImage(s['avatar'] as String),
                            backgroundColor: kSurface,
                          ),
                        ),
                      ),
                    ),
                    if (isYou)
                      Positioned(
                        bottom: 0,
                        right: 0,
                        child: Container(
                          width: 22,
                          height: 22,
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [kAccent, kAccent2],
                            ),
                            shape: BoxShape.circle,
                            border: Border.all(color: kBg, width: 2),
                          ),
                          child: const Icon(
                            Icons.add,
                            size: 14,
                            color: Color(0xFF09090B),
                          ),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  s['name'] as String,
                  style: GoogleFonts.inter(
                    fontSize: 11,
                    color: Colors.white70,
                    fontWeight: FontWeight.w400,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildBottomNav(BuildContext context) {
    return ClipRRect(
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 28, sigmaY: 28),
        child: Container(
          decoration: BoxDecoration(
            color: kBg.withOpacity(0.78),
            border: Border(
              top: BorderSide(color: kBorder.withOpacity(0.4), width: 0.3),
            ),
          ),
          child: SafeArea(
            child: SizedBox(
              height: 54,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _NavIcon(
                    icon: Icons.home_filled,
                    active: _activeNav == 0,
                    onTap: () => setState(() => _activeNav = 0),
                  ),
                  _NavIcon(
                    icon: Icons.search,
                    active: _activeNav == 1,
                    onTap: () => setState(() => _activeNav = 1),
                  ),
                  _NavIcon(
                    icon: Icons.add_box_outlined,
                    active: false,
                    onTap:
                        () => Navigator.push(
                          context,
                          MaterialPageRoute(builder: (_) => const UploadScreen()),
                        ),
                  ),
                  _NavIcon(
                    icon: Icons.movie_outlined,
                    active: _activeNav == 3,
                    onTap: () => setState(() => _activeNav = 3),
                  ),
                  _NavIcon(
                    iconWidget: Container(
                      padding: const EdgeInsets.all(1.5),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: _activeNav == 4 ? kAccent : Colors.transparent,
                          width: 1.5,
                        ),
                      ),
                      child: CircleAvatar(
                        radius: 12,
                        backgroundImage: const NetworkImage(
                          'https://i.pravatar.cc/150?img=5',
                        ),
                        backgroundColor: kSurface,
                      ),
                    ),
                    active: _activeNav == 4,
                    onTap: () => setState(() => _activeNav = 4),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _NavIcon extends StatelessWidget {
  final IconData? icon;
  final Widget? iconWidget;
  final bool active;
  final VoidCallback onTap;

  const _NavIcon({
    this.icon,
    this.iconWidget,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            iconWidget ??
                Icon(icon, color: active ? Colors.white : kMuted, size: 26),
            if (active && iconWidget == null) ...[
              const SizedBox(height: 4),
              Container(
                width: 4,
                height: 4,
                decoration: const BoxDecoration(
                  color: kAccent,
                  shape: BoxShape.circle,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _PostCard extends StatelessWidget {
  final Map<String, dynamic> post;
  final bool isLiked;
  final VoidCallback onLike;

  const _PostCard({
    required this.post,
    required this.isLiked,
    required this.onLike,
  });

  @override
  Widget build(BuildContext context) {
    final bool safe = post['safe'] as bool? ?? true;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(2),
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [
                      Color(0xFFC9A84C),
                      Color(0xFFE8C86A),
                      Color(0xFFF0D78C),
                    ],
                    begin: Alignment.bottomLeft,
                    end: Alignment.topRight,
                  ),
                ),
                child: Container(
                  padding: const EdgeInsets.all(1.5),
                  decoration: const BoxDecoration(
                    shape: BoxShape.circle,
                    color: kBg,
                  ),
                  child: CircleAvatar(
                    radius: 17,
                    backgroundImage: NetworkImage(post['avatar'] as String),
                    backgroundColor: kSurface,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          post['user'] as String,
                          style: GoogleFonts.inter(
                            fontWeight: FontWeight.w600,
                            fontSize: 13.5,
                            color: Colors.white,
                          ),
                        ),
                        if (post['verified'] as bool)
                          Padding(
                            padding: const EdgeInsets.only(left: 4),
                            child: ShaderMask(
                              shaderCallback:
                                  (b) => const LinearGradient(
                                    colors: [kAccent, kAccent2],
                                  ).createShader(b),
                              child: const Icon(
                                Icons.verified,
                                color: Colors.white,
                                size: 14,
                              ),
                            ),
                          ),
                      ],
                    ),
                    Text(
                      post['time'] as String,
                      style: GoogleFonts.inter(fontSize: 11, color: kMuted),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
                decoration: BoxDecoration(
                  color:
                      safe
                          ? const Color(0xFF22c55e).withOpacity(0.08)
                          : const Color(0xFFef4444).withOpacity(0.08),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      safe ? Icons.shield : Icons.shield_outlined,
                      size: 10,
                      color:
                          safe
                              ? const Color(0xFF4ade80)
                              : const Color(0xFFf87171),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      safe ? 'Safe' : 'Flagged',
                      style: GoogleFonts.inter(
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        color:
                            safe
                                ? const Color(0xFF4ade80)
                                : const Color(0xFFf87171),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.more_horiz, color: Colors.white38, size: 20),
            ],
          ),
        ),
        Image.network(
          post['image'] as String,
          width: double.infinity,
          height: 380,
          fit: BoxFit.cover,
          loadingBuilder:
              (ctx, child, prog) =>
                  prog == null
                      ? child
                      : Container(
                        width: double.infinity,
                        height: 380,
                        color: kSurface,
                        child: Center(
                          child: SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                              color: kAccent.withOpacity(0.6),
                              strokeWidth: 1.5,
                            ),
                          ),
                        ),
                      ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
          child: Row(
            children: [
              GestureDetector(
                onTap: onLike,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 200),
                  transitionBuilder: (child, anim) {
                    return ScaleTransition(scale: anim, child: child);
                  },
                  child: Icon(
                    isLiked ? Icons.favorite : Icons.favorite_border,
                    color: isLiked ? const Color(0xFFDC2626) : Colors.white,
                    size: 26,
                    key: ValueKey(isLiked),
                  ),
                ),
              ),
              const SizedBox(width: 16),
              const Icon(
                Icons.chat_bubble_outline,
                color: Colors.white,
                size: 23,
              ),
              const SizedBox(width: 16),
              const Icon(Icons.send_outlined, color: Colors.white, size: 23),
              const Spacer(),
              const Icon(Icons.bookmark_border, color: Colors.white, size: 23),
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 14),
          child: Text(
            '${isLiked ? "You and " : ""}${post["likes"]} likes',
            style: GoogleFonts.inter(
              fontWeight: FontWeight.w600,
              fontSize: 13,
              color: Colors.white,
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 4, 14, 2),
          child: RichText(
            text: TextSpan(
              children: [
                TextSpan(
                  text: '${post["user"]}  ',
                  style: GoogleFonts.inter(
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                    color: Colors.white,
                  ),
                ),
                TextSpan(
                  text: post['caption'] as String,
                  style: GoogleFonts.inter(
                    fontSize: 13,
                    color: Colors.white.withOpacity(0.85),
                  ),
                ),
              ],
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(14, 4, 14, 14),
          child: Text(
            'View all ${post["comments"]} comments',
            style: GoogleFonts.inter(fontSize: 12.5, color: kMuted),
          ),
        ),
        Divider(color: kBorder.withOpacity(0.4), height: 0.3, thickness: 0.3),
      ],
    );
  }
}
