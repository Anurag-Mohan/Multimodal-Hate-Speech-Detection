import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'upload_screen.dart';

const kBg = Color(0xFF000000);
const kCard = Color(0xFF111111);
const kSurface = Color(0xFF1A1A1A);
const kBorder = Color(0xFF262626);
const kAccent = Color(0xFF00f2fe);
const kAccent2 = Color(0xFF4facfe);
const kMuted = Color(0xFF8e8e8e);

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
              child: Divider(color: kBorder, thickness: 0.5, height: 1),
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
        bottomNavigationBar: _buildBottomNav(context),
      ),
    );
  }

  SliverAppBar _buildAppBar(BuildContext context) {
    return SliverAppBar(
      pinned: true,
      backgroundColor: kBg,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      title: ShaderMask(
        shaderCallback:
            (bounds) => const LinearGradient(
              colors: [kAccent, kAccent2, Color(0xFFa18cd1)],
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
          icon: const Icon(Icons.add_box_outlined, size: 26),
          onPressed:
              () => Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const UploadScreen()),
              ),
          color: Colors.white,
          tooltip: 'New Post',
        ),
        Stack(
          children: [
            IconButton(
              icon: const Icon(Icons.favorite_border, size: 26),
              onPressed: () {},
              color: Colors.white,
            ),
            Positioned(
              right: 8,
              top: 8,
              child: Container(
                width: 8,
                height: 8,
                decoration: const BoxDecoration(
                  color: Color(0xFFff3b5c),
                  shape: BoxShape.circle,
                ),
              ),
            ),
          ],
        ),
        IconButton(
          icon: const Icon(Icons.send_outlined, size: 26),
          onPressed: () {},
          color: Colors.white,
        ),
        const SizedBox(width: 4),
      ],
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(0.5),
        child: Divider(color: kBorder, height: 0.5, thickness: 0.5),
      ),
    );
  }

  Widget _buildStoriesRow() {
    return SizedBox(
      height: 106,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        itemCount: _stories.length,
        itemBuilder: (ctx, i) {
          final s = _stories[i];
          final isYou = s['isYou'] as bool;
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 6),
            child: Column(
              children: [
                Stack(
                  alignment: Alignment.center,
                  children: [
                    Container(
                      width: 68,
                      height: 68,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        gradient:
                            isYou
                                ? null
                                : const LinearGradient(
                                  colors: [
                                    Color(0xFFf58529),
                                    Color(0xFFdd2a7b),
                                    Color(0xFF8134af),
                                    Color(0xFF515bd4),
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
                        child: CircleAvatar(
                          backgroundImage: NetworkImage(s['avatar'] as String),
                          backgroundColor: kSurface,
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
                          decoration: const BoxDecoration(
                            color: kAccent,
                            shape: BoxShape.circle,
                          ),
                          child: const Icon(
                            Icons.add,
                            size: 16,
                            color: Colors.black,
                          ),
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  s['name'] as String,
                  style: GoogleFonts.inter(
                    fontSize: 11,
                    color: Colors.white,
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
    return Container(
      decoration: BoxDecoration(
        color: kBg,
        border: Border(top: BorderSide(color: kBorder, width: 0.5)),
      ),
      child: SafeArea(
        child: SizedBox(
          height: 52,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _NavIcon(icon: Icons.home, active: true, onTap: () {}),
              _NavIcon(icon: Icons.search, active: false, onTap: () {}),
              _NavIcon(
                icon: Icons.add_box_outlined,
                active: false,
                onTap:
                    () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const UploadScreen()),
                    ),
              ),
              _NavIcon(icon: Icons.movie_outlined, active: false, onTap: () {}),
              _NavIcon(
                iconWidget: CircleAvatar(
                  radius: 13,
                  backgroundImage: const NetworkImage(
                    'https://i.pravatar.cc/150?img=5',
                  ),
                  backgroundColor: kSurface,
                ),
                active: false,
                onTap: () {},
              ),
            ],
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
        padding: const EdgeInsets.all(10),
        child:
            iconWidget ??
            Icon(icon, color: active ? Colors.white : kMuted, size: 27),
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
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(2),
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [
                      Color(0xFFf58529),
                      Color(0xFFdd2a7b),
                      Color(0xFF8134af),
                    ],
                    begin: Alignment.bottomLeft,
                    end: Alignment.topRight,
                  ),
                ),
                child: CircleAvatar(
                  radius: 17,
                  backgroundImage: NetworkImage(post['avatar'] as String),
                  backgroundColor: kSurface,
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
                          const Padding(
                            padding: EdgeInsets.only(left: 4),
                            child: Icon(
                              Icons.verified,
                              color: kAccent,
                              size: 14,
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
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color:
                      safe ? const Color(0xFF0d2e1a) : const Color(0xFF2e0d0d),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color:
                        safe
                            ? const Color(0xFF22c55e)
                            : const Color(0xFFef4444),
                    width: 0.8,
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      safe ? Icons.shield : Icons.shield_outlined,
                      size: 11,
                      color:
                          safe
                              ? const Color(0xFF22c55e)
                              : const Color(0xFFef4444),
                    ),
                    const SizedBox(width: 3),
                    Text(
                      safe ? 'Safe' : 'Flagged',
                      style: GoogleFonts.inter(
                        fontSize: 10,
                        fontWeight: FontWeight.w600,
                        color:
                            safe
                                ? const Color(0xFF22c55e)
                                : const Color(0xFFef4444),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              const Icon(Icons.more_horiz, color: Colors.white, size: 20),
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
                        child: const Center(
                          child: CircularProgressIndicator(
                            color: kAccent,
                            strokeWidth: 1.5,
                          ),
                        ),
                      ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Row(
            children: [
              GestureDetector(
                onTap: onLike,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 200),
                  child: Icon(
                    isLiked ? Icons.favorite : Icons.favorite_border,
                    color: isLiked ? const Color(0xFFff3b5c) : Colors.white,
                    size: 26,
                    key: ValueKey(isLiked),
                  ),
                ),
              ),
              const SizedBox(width: 16),
              const Icon(
                Icons.chat_bubble_outline,
                color: Colors.white,
                size: 24,
              ),
              const SizedBox(width: 16),
              const Icon(Icons.send_outlined, color: Colors.white, size: 24),
              const Spacer(),
              const Icon(Icons.bookmark_border, color: Colors.white, size: 24),
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
                  style: GoogleFonts.inter(fontSize: 13, color: Colors.white),
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
        Divider(color: kBorder, height: 0.5, thickness: 0.5),
      ],
    );
  }
}
